module pernix_compression
    use iso_c_binding, only : c_double, c_float, c_int, c_int8_t, c_int32_t, c_ptr
    use pernix_dispatcher, only : PERNIX_BACKEND_AUTO, PERNIX_BACKEND_FALLBACK, PERNIX_STATUS_OK
    implicit none

    interface
        function pernix_compress_block_f32(dispatcher, bit_width, block_size, input_ptr, scale, output_ptr) &
                bind(C, name = "pernix_compress_block_f32") result(status)
            import :: c_float, c_int, c_int8_t, c_int32_t, c_ptr
            type(c_ptr), value :: dispatcher
            integer(c_int8_t), value :: bit_width
            integer(c_int32_t), value :: block_size
            type(c_ptr), value :: input_ptr
            real(c_float), value :: scale
            type(c_ptr), value :: output_ptr
            integer(c_int) :: status
        end function

        function pernix_compress_blocks_f32(dispatcher, bit_width, block_size, input_ptr, scale, output_ptr, blocks) &
                bind(C, name = "pernix_compress_blocks_f32") result(status)
            import :: c_float, c_int, c_int8_t, c_int32_t, c_ptr
            type(c_ptr), value :: dispatcher
            integer(c_int8_t), value :: bit_width
            integer(c_int32_t), value :: block_size
            type(c_ptr), value :: input_ptr
            real(c_float), value :: scale
            type(c_ptr), value :: output_ptr
            integer(c_int32_t), value :: blocks
            integer(c_int) :: status
        end function

        function pernix_compress_block_f64(dispatcher, bit_width, block_size, input_ptr, scale, output_ptr) &
                bind(C, name = "pernix_compress_block_f64") result(status)
            import :: c_double, c_int, c_int8_t, c_int32_t, c_ptr
            type(c_ptr), value :: dispatcher
            integer(c_int8_t), value :: bit_width
            integer(c_int32_t), value :: block_size
            type(c_ptr), value :: input_ptr
            real(c_double), value :: scale
            type(c_ptr), value :: output_ptr
            integer(c_int) :: status
        end function

        function pernix_compress_blocks_f64(dispatcher, bit_width, block_size, input_ptr, scale, output_ptr, blocks) &
                bind(C, name = "pernix_compress_blocks_f64") result(status)
            import :: c_double, c_int, c_int8_t, c_int32_t, c_ptr
            type(c_ptr), value :: dispatcher
            integer(c_int8_t), value :: bit_width
            integer(c_int32_t), value :: block_size
            type(c_ptr), value :: input_ptr
            real(c_double), value :: scale
            type(c_ptr), value :: output_ptr
            integer(c_int32_t), value :: blocks
            integer(c_int) :: status
        end function
    end interface
end module pernix_compression
