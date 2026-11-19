#ifndef POOL_HPP
#define POOL_HPP

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

class ThreadPool
{
public:
  using Task = std::function<void()>;

  explicit ThreadPool(std::size_t thread_count);
  ~ThreadPool();

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  template<class F, class... Args>
  auto submit(F&& function, Args&&... args)
    -> std::future<typename std::result_of<
      typename std::decay<F>::type(typename std::decay<Args>::type...)>::type>
  {
    using Result = typename std::result_of<
      typename std::decay<F>::type(typename std::decay<Args>::type...)>::type;

    auto callable = [fn = std::forward<F>(function),
                     values = std::make_tuple(std::forward<Args>(args)...)]() mutable -> Result
    {
      return invokeTuple(fn, values, std::index_sequence_for<Args...>());
    };

    auto packaged_task = std::make_shared<std::packaged_task<Result()>>(std::move(callable));
    std::future<Result> result = packaged_task->get_future();

    {
      std::lock_guard<std::mutex> state_lock(state_mutex_);
      if (stopping_)
      {
        throw std::runtime_error("Cannot submit to a stopped thread pool");
      }

      const std::size_t worker_id = leastBusyWorker();
      Worker& worker = *workers_[worker_id];
      {
        std::lock_guard<std::mutex> queue_lock(worker.mutex);
        worker.tasks.emplace_back([packaged_task]()
        {
          (*packaged_task)();
        });
        worker.queued.fetch_add(1, std::memory_order_relaxed);
      }
      ++pending_tasks_;
    }

    work_available_.notify_all();
    return result;
  }

  void waitIdle();
  void pause();
  void resume();
  void stop();

private:
  struct Worker
  {
    std::deque<Task> tasks;
    std::mutex mutex;
    std::atomic<std::size_t> queued{0};
  };

  std::vector<std::unique_ptr<Worker>> workers_;
  std::vector<std::thread> threads_;

  mutable std::mutex state_mutex_;
  std::condition_variable work_available_;
  std::condition_variable idle_;
  std::size_t pending_tasks_ = 0;
  std::size_t active_tasks_ = 0;
  bool stopping_ = false;
  bool paused_ = false;

  static constexpr std::size_t batch_size_ = 4;

  void workerLoop(std::size_t worker_id);
  bool takeLocal(std::size_t worker_id, std::vector<Task>& batch);
  bool steal(std::size_t worker_id, std::vector<Task>& batch);
  void runBatch(std::vector<Task>& batch);
  std::size_t leastBusyWorker() const;

  template<class F, class Tuple, std::size_t... Indices>
  static auto invokeTuple(F& function, Tuple& values, std::index_sequence<Indices...>)
    -> decltype(function(std::move(std::get<Indices>(values))...))
  {
    return function(std::move(std::get<Indices>(values))...);
  }
};

#endif
