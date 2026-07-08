module pernix_decompression
    use iso_c_binding, only : c_bool, c_double, c_float, c_int, c_int8_t, c_int32_t, c_ptr
    implicit none

    interface
        function pernix_decompress_block_f32(backend, bit_width, block_size, input_ptr, scale, output_ptr, sign_values) &
                bind(C, name = "pernix_decompress_block_f32") result(status)
            import :: c_bool, c_float, c_int, c_int8_t, c_int32_t, c_ptr
            integer(c_int), value :: backend
            integer(c_int8_t), value :: bit_width
            integer(c_int32_t), value :: block_size
            type(c_ptr), value :: input_ptr
            real(c_float), value :: scale
            type(c_ptr), value :: output_ptr
            logical(c_bool), value :: sign_values
            integer(c_int) :: status
        end function

        function pernix_decompress_blocks_f32(backend, bit_width, block_size, input_ptr, scale, output_ptr, blocks, &
                sign_values) bind(C, name = "pernix_decompress_blocks_f32") result(status)
            import :: c_bool, c_float, c_int, c_int8_t, c_int32_t, c_ptr
            integer(c_int), value :: backend
            integer(c_int8_t), value :: bit_width
            integer(c_int32_t), value :: block_size
            type(c_ptr), value :: input_ptr
            real(c_float), value :: scale
            type(c_ptr), value :: output_ptr
            integer(c_int32_t), value :: blocks
            logical(c_bool), value :: sign_values
            integer(c_int) :: status
        end function

        function pernix_decompress_block_f64(backend, bit_width, block_size, input_ptr, scale, output_ptr, sign_values) &
                bind(C, name = "pernix_decompress_block_f64") result(status)
            import :: c_bool, c_double, c_int, c_int8_t, c_int32_t, c_ptr
            integer(c_int), value :: backend
            integer(c_int8_t), value :: bit_width
            integer(c_int32_t), value :: block_size
            type(c_ptr), value :: input_ptr
            real(c_double), value :: scale
            type(c_ptr), value :: output_ptr
            logical(c_bool), value :: sign_values
            integer(c_int) :: status
        end function

        function pernix_decompress_blocks_f64(backend, bit_width, block_size, input_ptr, scale, output_ptr, blocks, &
                sign_values) bind(C, name = "pernix_decompress_blocks_f64") result(status)
            import :: c_bool, c_double, c_int, c_int8_t, c_int32_t, c_ptr
            integer(c_int), value :: backend
            integer(c_int8_t), value :: bit_width
            integer(c_int32_t), value :: block_size
            type(c_ptr), value :: input_ptr
            real(c_double), value :: scale
            type(c_ptr), value :: output_ptr
            integer(c_int32_t), value :: blocks
            logical(c_bool), value :: sign_values
            integer(c_int) :: status
        end function
    end interface
end module pernix_decompression
