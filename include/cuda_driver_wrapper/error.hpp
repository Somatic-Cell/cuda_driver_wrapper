#pragma once

#include <cuda.h>

#include <source_location>
#include <stdexcept>
#include <string>

namespace cuda_driver_wrapper
{

// 失敗の情報を保持する例外型．エラー処理を実行する管理クラスではない
// what() の診断に使えて，元の CUresult を文字列解析なしで取得可能
class CudaError final : public std::runtime_error
{
public:
    CudaError(CUresult result, const std::string& message);

    [[nodiscard]]
    CUresult result() const noexcept;
private:
    CUresult result_;
};


// 成功時には何もせずに，失敗時に診断情報を含む CudaError を送る
// 同期や文字列の
void check_cuda(
    CUresult result,
    const char* operation,                                          // 呼び出し中だけ参照する文字列．nullptr も受け付ける
    std::source_location location = std::source_location::current()
);

// 後始末用の関数
// 成功なら true, 失敗なら stderr に 報告を試みた後に false を返す
// リソースの開放・再試行・例外の送出はしない．また，出力の成功までは保証しない
[[nodiscard]]
bool report_cuda_cleanup_result(
    CUresult result,
    const char* operation,
    std::source_location location = std::source_location::current()
) noexcept;


} // namespace cuda_driver_wrapper