#pragma once

#include <cuda.h>
#include <source_location>

namespace cuda_driver_wrapper
{
class Stream;   // 前方宣言

// CUevent を一つ所有し，context は借用する．自動で retain / 同期は行わない
// 同じオブジェクトの record / 使用と close / move /destruct は呼び出し側で直列化
class Event final
{
public:
    explicit Event(
        CUcontext context,
        unsigned int flags = CU_EVENT_DISABLE_TIMING,
        std::source_location location = std::source_location::current()
    );

    ~Event() noexcept;

    // コピーの禁止
    Event(const Event&) = delete;               // コピーコンストラクタの禁止
    Event& operator=(const Event&) = delete;    // コピー代入の禁止

    // ムーブ
    Event(Event&& other) noexcept;
    Event& operator=(Event&& other);

    void close(std::source_location location = std::source_location::current());

    // 借用ハンドル
    [[nodiscard]] CUevent handle() const noexcept {return event_; }
    [[nodiscard]] CUcontext context() const noexcept {return context_; }
    [[nodiscard]] bool is_valid() const noexcept {return event_ != nullptr; }

private:
    void close_noexcept() noexcept;

    CUcontext context_ = nullptr;
    CUevent event_ = nullptr;
};

void record_event(
    Event& event,
    const Stream& stream,
    std::source_location location = std::source_location::current()
);

void synchronize_event(
    const Event& event,
    std::source_location location = std::source_location::current()
);

[[nodiscard]] bool is_event_ready(
    const Event& event,
    std::source_location location = std::source_location::current()
);

void stream_wait_event(
    const Stream& stream,
    const Event& event,
    std::source_location location = std::source_location::current()
);

[[nodiscard]] float elapsed_time_ms(
    const Event& start,
    const Event& end,
    std::source_location location = std::source_location::current()
);

} // namespace cuda_driver_wrapper