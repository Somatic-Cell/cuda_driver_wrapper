#pragma once

#include <cuda.h>

#include <source_location>
#include <thread>

namespace cuda_driver_wrapper
{

[[nodiscard]]
CUcontext get_current_context(
    std::source_location location = std::source_location::current()
);


// 指定したデバイスの primary context に対する retain の参照を1つ保持する
// context 全体を独占所有するわけではない
class PrimaryContext final
{
public:
    explicit PrimaryContext(
        CUdevice device,
        std::source_location location = std::source_location::current()
    );

    // release の失敗は報告するが，例外は送出しない
    ~PrimaryContext() noexcept;

    // コピーの禁止
    PrimaryContext(const PrimaryContext&) = delete;                 // コピーコンストラクタの禁止
    PrimaryContext& operator=(const PrimaryContext&) = delete;      // コピー代入の禁止
    
    // move の禁止
    PrimaryContext(PrimaryContext&&) = delete;                      // ムーブコンストラクタの禁止
    PrimaryContext& operator=(PrimaryContext&&) = delete;           // ムーブ代入の禁止

    [[nodiscard]]
    CUdevice device() const noexcept {return device_;}

    // この関数を実行しただけでは current にならない
    [[nodiscard]]
    CUcontext handle() const noexcept {return context_;}

private:
    CUdevice device_;
    CUcontext context_ = nullptr;
};

// 有効な CUcontext を借用して，構築時に一度 push, 復元時に一度 pop する
// どのコンテキストでも使用可能
class [[nodiscard]] ScopedCurrentContext final
{
public:
    // nullptr は std::invalid_argument として拒否
    explicit ScopedCurrentContext(
        CUcontext context,
        std::source_location location = std::source_location::current()
    );

    ~ScopedCurrentContext() noexcept;

    // コピーの禁止
    ScopedCurrentContext(const ScopedCurrentContext&) = delete;
    ScopedCurrentContext& operator=(const ScopedCurrentContext&) = delete;

    // move の禁止
    ScopedCurrentContext(ScopedCurrentContext&&) = delete;
    ScopedCurrentContext& operator=(ScopedCurrentContext&&) = delete;

    void restore(
        std::source_location location = std::source_location::current()
    );

    [[nodiscard]]
    bool is_active() const noexcept {return active_; }

private:
    CUcontext context_;
    std::thread::id thread_id_;
    bool active_ = false;
};


} // namespace cuda_driver_wrapper