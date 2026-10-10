// ------------------------------------
// CUDA Stream: GPU に投入する仕事を順序付ける仕組み
// （ある意味）CPU のスレッドに似ているが，GPU に投入する順序を表すオブジェクト
// ------------------------------------
#pragma once

#include <cuda.h>
#include <source_location>

namespace cuda_driver_wrapper
{

// ストリームを識別する CUStream と，所属するコンテキストの CUContext を保持する
class Stream final
{
public:
    explicit Stream(
        CUcontext context,
        unsigned int flags = CU_STREAM_NON_BLOCKING,
        std::source_location location = std::source_location::current()
    );

    ~Stream() noexcept;

    // コピーの禁止
    Stream(const Stream&) = delete;             // コピーコンストラクタの禁止
    Stream& operator=(const Stream&) = delete;  // コピー代入の禁止

    // ムーブ
    // このクラスは CUDA のストリームを作り直したり，
    // 別の GPU はコンテキストに移動したりしないため，
    // ムーブを許す
    Stream(Stream&& other) noexcept;
    Stream& operator=(Stream&& other);      // 古い stream の close が失敗する可能性を考え，ムーブ代入は noexcept にしない

    // 破棄を要求するが，ストリームに投入済みの処理に対して完了待ちはしない
    // 成功すると空になり，もともと空のオブジェクトで呼んでも何もしない
    void close(std::source_location location = std::source_location::current());

    // 借用ハンドル．呼び出し側で destroy しない．API の current の設定は呼び出し側の責任
    [[nodiscard]] CUstream handle() const noexcept {return stream_; }
    [[nodiscard]] CUcontext context() const noexcept {return context_; }
    [[nodiscard]] bool is_valid() const noexcept {return stream_ != nullptr; }

private:
    void close_noexcept() noexcept;

    CUcontext context_ = nullptr;
    CUstream stream_ = nullptr;
};

// ホスト側でこのストリームに投入した処理の完了を待つ
void synchronize_stream(
    const Stream& stream,
    std::source_location location = std::source_location::current()
);

// CUDA_SUCCESS なら true, CUDA_ERROR_NOT_READY なら false
// その他の CUA Error はCudaError として送出
// close / move 後の空の stream は拒否
[[nodiscard]] bool is_stream_ready(
    const Stream& stream,
    std::source_location location = std::source_location::current()
);

} // namespace cuda_driver_pointer