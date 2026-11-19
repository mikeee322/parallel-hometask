#include "pool.hpp"

#include <algorithm>

constexpr std::size_t ThreadPool::batch_size_;

ThreadPool::ThreadPool(std::size_t thread_count)
{
  if (thread_count == 0)
  {
    thread_count = 1;
  }

  workers_.reserve(thread_count);
  threads_.reserve(thread_count);

  for (std::size_t i = 0; i < thread_count; ++i)
  {
    workers_.push_back(std::make_unique<Worker>());
  }

  try
  {
    for (std::size_t i = 0; i < thread_count; ++i)
    {
      threads_.emplace_back(&ThreadPool::workerLoop, this, i);
    }
  }
  catch (...)
  {
    {
      std::lock_guard<std::mutex> state_lock(state_mutex_);
      stopping_ = true;
    }
    work_available_.notify_all();

    for (std::thread& thread : threads_)
    {
      if (thread.joinable())
      {
        thread.join();
      }
    }
    throw;
  }
}

ThreadPool::~ThreadPool()
{
  stop();
}

std::size_t ThreadPool::leastBusyWorker() const
{
  std::size_t best = 0;

  for (std::size_t i = 1; i < workers_.size(); ++i)
  {
    if (workers_[i]->queued.load(std::memory_order_relaxed) <
        workers_[best]->queued.load(std::memory_order_relaxed))
    {
      best = i;
    }
  }

  return best;
}

bool ThreadPool::takeLocal(std::size_t worker_id, std::vector<Task>& batch)
{
  Worker& worker = *workers_[worker_id];
  std::size_t taken = 0;

  {
    std::lock_guard<std::mutex> queue_lock(worker.mutex);
    taken = std::min(batch_size_, worker.tasks.size());

    for (std::size_t i = 0; i < taken; ++i)
    {
      batch.push_back(std::move(worker.tasks.back()));
      worker.tasks.pop_back();
    }

    worker.queued.fetch_sub(taken, std::memory_order_relaxed);
  }

  if (taken == 0)
  {
    return false;
  }

  {
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    pending_tasks_ -= taken;
    active_tasks_ += taken;
  }

  return true;
}

bool ThreadPool::steal(std::size_t worker_id, std::vector<Task>& batch)
{
  const std::size_t worker_count = workers_.size();
  if (worker_count < 2)
  {
    return false;
  }

  for (std::size_t offset = 1; offset < worker_count; ++offset)
  {
    const std::size_t victim_id = (worker_id + offset) % worker_count;
    Worker& victim = *workers_[victim_id];
    std::size_t taken = 0;

    {
      std::lock_guard<std::mutex> queue_lock(victim.mutex);
      if (victim.tasks.empty())
      {
        continue;
      }

      // Take up to half of the victim's oldest tasks, with a batch size limit.
      taken = std::min(batch_size_, std::max<std::size_t>(1, victim.tasks.size() / 2));

      for (std::size_t i = 0; i < taken; ++i)
      {
        batch.push_back(std::move(victim.tasks.front()));
        victim.tasks.pop_front();
      }

      victim.queued.fetch_sub(taken, std::memory_order_relaxed);
    }

    {
      std::lock_guard<std::mutex> state_lock(state_mutex_);
      pending_tasks_ -= taken;
      active_tasks_ += taken;
    }

    return true;
  }

  return false;
}

void ThreadPool::runBatch(std::vector<Task>& batch)
{
  for (Task& task : batch)
  {
    try
    {
      task();
    }
    catch (...)
    {
      // packaged_task stores exceptions from submitted functions in their futures.
    }
  }

  {
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    active_tasks_ -= batch.size();
    if (pending_tasks_ == 0 && active_tasks_ == 0)
    {
      idle_.notify_all();
    }
  }

  batch.clear();
}

void ThreadPool::workerLoop(std::size_t worker_id)
{
  std::vector<Task> batch;
  batch.reserve(batch_size_);

  while (true)
  {
    {
      std::unique_lock<std::mutex> state_lock(state_mutex_);
      work_available_.wait(state_lock, [this]()
      {
        return stopping_ || (!paused_ && pending_tasks_ > 0);
      });

      if (stopping_ && pending_tasks_ == 0)
      {
        return;
      }

      if (paused_)
      {
        continue;
      }
    }

    batch.clear();
    if (takeLocal(worker_id, batch) || steal(worker_id, batch))
    {
      runBatch(batch);
    }
  }
}

void ThreadPool::waitIdle()
{
  std::unique_lock<std::mutex> state_lock(state_mutex_);
  idle_.wait(state_lock, [this]()
  {
    return pending_tasks_ == 0 && active_tasks_ == 0;
  });
}

void ThreadPool::pause()
{
  std::lock_guard<std::mutex> state_lock(state_mutex_);
  if (!stopping_)
  {
    paused_ = true;
  }
}

void ThreadPool::resume()
{
  {
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    paused_ = false;
  }

  work_available_.notify_all();
}

void ThreadPool::stop()
{
  {
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    stopping_ = true;
    paused_ = false;
  }

  work_available_.notify_all();

  for (std::thread& thread : threads_)
  {
    if (thread.joinable())
    {
      thread.join();
    }
  }

  threads_.clear();
}
