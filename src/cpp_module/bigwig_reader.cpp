#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> // For std::vector, std::string, to bind STL types to Python with ease
// #include <libbigwig/bigWig.h> // libBigWig header. Read and write BigWig files.
#include <bigWig.h> // libBigWig header. Read and write BigWig files.
#include <stdexcept> // For std::runtime_error
#include <algorithm> // For std::max. Useful function.
#include <vector> // For std::vector
#include <cmath> // For std::isnan, NAN

namespace py = pybind11; // Declare to use "pybind11" as "py" 

class BigWigReader {
public:
    bigWigFile_t *bw; // BigWig file handle

    // Constructor: Open a BigWig file
    BigWigReader(const std::string& bw_path) {
        bw = bwOpen(bw_path.c_str(), NULL, "r");
        if (!bw) {
            throw std::runtime_error("Failed to open BigWig file: " + bw_path);
        }
    }

    // Destructor: Close the BigWig file
    ~BigWigReader() {
        if (bw) {
            bwClose(bw);
            bw = nullptr; // Set pointer to NULL to prevent double free
        }
    }

    // Function definition to get values from the BigWig file
    // Return type: tuple(values, actual_start_coords, actual_end_coords) equivalent in Python
    py::tuple get_values(const std::string& chrom_str, int start_coords, int end_coords) {
        if (!bw) {
            throw std::runtime_error("BigWig file is not open.");
        }

        int chrom_length = 0;
        bool chrom_found = false;
        for (int i = 0; i < bw->cl->nKeys; ++i) {
            if (chrom_str == bw->cl->chrom[i]) {
                chrom_length = bw->cl->len[i];
                chrom_found = true;
                break;
            }
        }

        // If the chromosome is not found or its length is zero, return a zero-length array
        if (!chrom_found || chrom_length == 0) {
            // To adapt to pyBigWig's behavior, return a zero-length array
            int requested_len = std::max(0, end_coords - start_coords);
            return py::make_tuple(py::array_t<float>(requested_len), start_coords, end_coords);
        }

        // Adjust start and end coordinates to be within the chromosome length
        int actual_start_coords = start_coords;
        int actual_end_coords = end_coords;

        if (actual_end_coords == -1 || actual_end_coords > chrom_length) {
            actual_end_coords = chrom_length;
        }
        if (actual_start_coords < 0) {
            actual_start_coords = 0;
        }
        
        // If the range is invalid (start >= end), return an empty array.
        // However, pyBigWig returns an array of length end_coords - start_coords if it is non-negative.
        int array_len = actual_end_coords - actual_start_coords;
        if (array_len < 0) { // If the length is negative exceptionally
            array_len = 0;
        }

        std::vector<float> signal_vector(array_len);

        if (array_len > 0) { // If there is a range to retrieve
            // float *values = bwReadValues(bw, chrom_str.c_str(), actual_start_coords, actual_end_coords, 0); // 0はdoMedian
            bwOverlappingIntervals_t* values = bwGetValues(bw, chrom_str.c_str(), actual_start_coords, actual_end_coords, 0); // 0: doMedian
            
            if (values) {
                for (int i = 0; i < array_len; ++i) {
                    signal_vector[i] = values->value[i]; // bwGetValues で取得した値をコピー
                }
                free(values); // bwReadValues で割り当てられたメモリを解放
            } else {
                // If no data is retrieved, fill with NaN
                for (int i = 0; i < array_len; ++i) {
                    signal_vector[i] = NAN;
                }
            }
        }
        
        // std::vector<float> -> Numpy array
        py::array_t<float> signal_array = py::cast(signal_vector);
        
        return py::make_tuple(signal_array, actual_start_coords, actual_end_coords);
    }
};

py::array_t<float> adjust_array_length(
    py::array_t<float> signal_array,
    int start, int end,
    int actual_start, int actual_end)
{
    // numpy.arrayを操作するためにバッファ情報を取得
    auto buf1 = signal_array.request();
    if (buf1.ndim != 1) {
        throw std::runtime_error("Input signal_array must be a 1-dimensional array.");
    }
    
    // 引数として渡されたnumpy arrayの長さ
    int original_length = buf1.shape[0];
    // 要求された新しい長さ
    int requested_length = end - start;

    // 長さが既に一致している場合は、コピーせずにそのまま返す
    if (requested_length == original_length) {
        return signal_array;
    }
    
    // 新しい配列をNaNで初期化
    py::array_t<float> new_array = py::array_t<float>(requested_length);
    auto buf2 = new_array.request();
    float *ptr_new = static_cast<float*>(buf2.ptr);
    std::fill(ptr_new, ptr_new + requested_length, NAN);

    // Pythonのスライス [actual_start - start : actual_end - start] に相当するC++のオフセットと長さを計算
    int copy_start_index = actual_start - start;
    int copy_end_index = actual_end - start;
    int copy_length = original_length; // コピーするデータの長さは、元の配列の長さ

    // コピーする範囲が新しい配列の境界内に収まっているか確認
    int new_array_start = std::max(0, copy_start_index);
    int new_array_end = std::min(requested_length, copy_end_index);
    int new_array_copy_length = new_array_end - new_array_start;

    if (new_array_copy_length > 0) {
        // 元の配列からコピーを開始する位置を計算
        int original_array_offset = std::max(0, start - actual_start);
        
        // Copy values rapidly using memory copy (std::copy)
        std::copy(static_cast<float*>(buf1.ptr) + original_array_offset,
                  static_cast<float*>(buf1.ptr) + original_array_offset + new_array_copy_length,
                  ptr_new + new_array_start);
    }
    
    return new_array;
}

// Define Python module
PYBIND11_MODULE(bigwig_cpp_reader, m) {
    m.doc() = "pybind11 BigWig reader module";

    // Open BigWigReader class as a Python class
    py::class_<BigWigReader>(m, "BigWigReader")
        .def(py::init<const std::string&>(), "Initialize BigWigReader with a BigWig file path.")
        .def("get_values", &BigWigReader::get_values,
             "Get signal values from the opened BigWig file for a specified genomic region.",
             py::arg("chrom_str"), py::arg("start_coords"), py::arg("end_coords") = -1); // Set default arguments
    
    m.def("adjust_array_length", &adjust_array_length,
       "Adjusts a signal array to a fixed length, padding with NaNs if necessary.",
       py::arg("signal_array"),
       py::arg("start"), py::arg("end"),
       py::arg("actual_start"), py::arg("actual_end"));
}
// Can get signal values from the opened BigWig file for a specified chromosome and coordinate range using the BigWigReader class from Python.