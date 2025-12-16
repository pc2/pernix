#include <gtest/gtest.h>

#include <libcompression/decompression.h>

int main(int argc, char* argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();

    // const UnpackingTestSet<8, true, 64> test_set;
    //
    // std::cout << "Unpacked data: ";
    // for (const auto& val : test_set.get_unpacked_data()) {
    //     std::cout << val << " ";
    // }
}
