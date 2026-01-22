module pernix_compression
    use iso_c_binding, only : c_int8_t, c_int32_t, c_float, c_ptr
    implicit none

    interface
        subroutine mm256_compress_block_bmi2(bit_width, input_ptr, scale, output_ptr) bind(C, name = "mm256_compress_block_bmi2_c")
            import :: c_int8_t, c_float, c_ptr
            integer(c_int8_t), value :: bit_width
            type(c_ptr), value :: input_ptr
            real(c_float), value :: scale
            type(c_ptr), value :: output_ptr
        end subroutine

        subroutine mm256_compress_block_avx2(bit_width, input_ptr, scale, output_ptr) bind(C, name = "mm256_compress_block_avx2_c")
            import :: c_int8_t, c_float, c_ptr
            integer(c_int8_t), value :: bit_width
            type(c_ptr), value :: input_ptr
            real(c_float), value :: scale
            type(c_ptr), value :: output_ptr
        end subroutine

        subroutine mm512_compress_block_avx512vbmi(bit_width, input_ptr, scale, output_ptr) bind(C, name = "mm512_compress_block_avx512vbmi_c")
            import :: c_int8_t, c_float, c_ptr
            integer(c_int8_t), value :: bit_width
            type(c_ptr), value :: input_ptr
            real(c_float), value :: scale
            type(c_ptr), value :: output_ptr
        end subroutine

        subroutine compress_block_fallback(bit_width, input_ptr, scale, output_ptr) bind(C, name = "compress_block_fallback_c")
            import :: c_int8_t, c_float, c_ptr
            integer(c_int8_t), value :: bit_width
            type(c_ptr), value :: input_ptr
            real(c_float), value :: scale
            type(c_ptr), value :: output_ptr
        end subroutine


        subroutine mm256_compress_blocks_bmi2(bit_width, input_ptr, scale, output_ptr, blocks) bind(C, name = "mm256_compress_blocks_bmi2_c")
            import :: c_int8_t, c_int32_t, c_float, c_ptr
            integer(c_int8_t), value :: bit_width
            type(c_ptr), value :: input_ptr
            real(c_float), value :: scale
            type(c_ptr), value :: output_ptr
            integer(c_int32_t), value :: blocks
        end subroutine

        subroutine mm256_compress_blocks_avx2(bit_width, input_ptr, scale, output_ptr, blocks) bind(C, name = "mm256_compress_blocks_avx2_c")
            import :: c_int8_t, c_int32_t, c_float, c_ptr
            integer(c_int8_t), value :: bit_width
            type(c_ptr), value :: input_ptr
            real(c_float), value :: scale
            type(c_ptr), value :: output_ptr
            integer(c_int32_t), value :: blocks
        end subroutine

        subroutine mm512_compress_blocks_avx512vbmi(bit_width, input_ptr, scale, output_ptr, blocks) bind(C, name = "mm512_compress_blocks_avx512vbmi_c")
            import :: c_int8_t, c_int32_t, c_float, c_ptr
            integer(c_int8_t), value :: bit_width
            type(c_ptr), value :: input_ptr
            real(c_float), value :: scale
            type(c_ptr), value :: output_ptr
            integer(c_int32_t), value :: blocks
        end subroutine

        subroutine compress_blocks_fallback(bit_width, input_ptr, scale, output_ptr, blocks) bind(C, name = "compress_blocks_fallback_c")
            import :: c_int8_t, c_int32_t, c_float, c_ptr
            integer(c_int8_t), value :: bit_width
            type(c_ptr), value :: input_ptr
            real(c_float), value :: scale
            type(c_ptr), value :: output_ptr
            integer(c_int32_t), value :: blocks
        end subroutine


    end interface

end module pernix_compression