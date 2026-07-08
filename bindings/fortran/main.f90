program pernix_fortran_roundtrip
    use iso_c_binding, only : c_bool, c_float, c_int, c_int8_t, c_int32_t, c_loc
    use pernix_compression
    use pernix_decompression
    implicit none

    integer(c_int8_t), parameter :: bit_width = 16_c_int8_t
    integer(c_int32_t), parameter :: block_size = 64_c_int32_t
    integer, parameter :: elements = (64 * 8) / 16
    real(c_float), target :: input(elements)
    real(c_float), target :: restored(elements)
    integer(c_int8_t), target :: compressed(block_size)
    real(c_float) :: bmax
    real(c_float) :: scale
    integer(c_int) :: status
    integer :: i

    do i = 1, elements
        input(i) = real(i - 17, c_float) * 0.125_c_float
    end do

    bmax = maxval(abs(input))
    scale = bmax / real((2 ** (int(bit_width) - 1)) - 1, c_float)

    status = pernix_compress_block_f32(PERNIX_BACKEND_FALLBACK, bit_width, block_size, c_loc(input), &
                                       1.0_c_float / scale, c_loc(compressed))
    if (status /= PERNIX_STATUS_OK) stop 1

    status = pernix_decompress_block_f32(PERNIX_BACKEND_FALLBACK, bit_width, block_size, c_loc(compressed), &
                                         scale, c_loc(restored), .true._c_bool)
    if (status /= PERNIX_STATUS_OK) stop 2

    do i = 1, elements
        if (abs(restored(i) - input(i)) > scale) stop 3
    end do
end program pernix_fortran_roundtrip
