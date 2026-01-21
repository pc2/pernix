module pernix
    use iso_c_binding, only : c_int8_t, c_int32_t, c_float, c_ptr
    implicit none

    interface
        subroutine mm256_decompress_block_bmi2_c(bit_width, input_ptr, scale, output_ptr) bind(C, name = "mm256_decompress_block_bmi2_c")
            import :: c_int8_t, c_float, c_ptr
            integer(c_int8_t), value :: bit_width
            type(c_ptr), value :: input_ptr
            real(c_float), value :: scale
            type(c_ptr), value :: output_ptr
        end subroutine

        subroutine mm256_decompress_block_avx2_c(bit_width, input_ptr, scale, output_ptr) bind(C, name = "mm256_decompress_block_avx2_c")
            import :: c_int8_t, c_float, c_ptr
            integer(c_int8_t), value :: bit_width
            type(c_ptr), value :: input_ptr
            real(c_float), value :: scale
            type(c_ptr), value :: output_ptr
        end subroutine

        subroutine decompress_block_fallback_c(bit_width, input_ptr, scale, output_ptr) bind(C, name = "decompress_block_fallback_c")
            import :: c_int8_t, c_float, c_ptr
            integer(c_int8_t), value :: bit_width
            type(c_ptr), value :: input_ptr
            real(c_float), value :: scale
            type(c_ptr), value :: output_ptr
        end subroutine


        subroutine mm256_decompress_blocks_bmi2_c(bit_width, input_ptr, scale, output_ptr, blocks) bind(C, name = "mm256_decompress_blocks_bmi2_c")
            import :: c_int8_t, c_int32_t, c_float, c_ptr
            integer(c_int8_t), value :: bit_width
            type(c_ptr), value :: input_ptr
            real(c_float), value :: scale
            type(c_ptr), value :: output_ptr
            integer(c_int32_t), value :: blocks
        end subroutine

        subroutine mm256_decompress_blocks_avx2_c(bit_width, input_ptr, scale, output_ptr, blocks) bind(C, name = "mm256_decompress_blocks_avx2_c")
            import :: c_int8_t, c_int32_t, c_float, c_ptr
            integer(c_int8_t), value :: bit_width
            type(c_ptr), value :: input_ptr
            real(c_float), value :: scale
            type(c_ptr), value :: output_ptr
            integer(c_int32_t), value :: blocks
        end subroutine

        subroutine decompress_blocks_fallback_c(bit_width, input_ptr, scale, output_ptr, blocks) bind(C, name = "decompress_blocks_fallback_c")
            import :: c_int8_t, c_int32_t, c_float, c_ptr
            integer(c_int8_t), value :: bit_width
            type(c_ptr), value :: input_ptr
            real(c_float), value :: scale
            type(c_ptr), value :: output_ptr
            integer(c_int32_t), value :: blocks
        end subroutine

    end interface

end module pernix