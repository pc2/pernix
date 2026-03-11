! Hello World Fortran program
program main
    use iso_c_binding, only : c_int8_t, c_int32_t, c_float, c_loc, c_ptr
    use iso_fortran_env, only : real64, int64
    use pernix_decompression
    use pernix_compression
    use omp_lib
    implicit none

    integer(c_int8_t) :: bit_width = 16_c_int8_t
    real(c_float) :: scale
    integer(c_int8_t), target :: input_data(512)
    real(c_float), target :: output_data(512)
    integer :: i, j, gb, iter
    real(real64) :: t0, t1, diff

    ! Initialize example data
    scale = 1.5_c_float
    output_data = 0.0_c_float

    iter = 50000000_int64
    t0 = omp_get_wtime()
    do i = 1_int64, iter
        do j = 1, size(input_data)
            input_data(j) = mod(i + j, 256_int64)
        end do

        ! call the C-binding Fortran subroutine using C pointers
        call decompress_block(bit_width, c_loc(input_data), scale, c_loc(output_data))
    end do
    t1 = omp_get_wtime()
    diff = t1 - t0
    gb = iter * (512_int64 + 1024_int64) / (1024_int64 * 1024_int64 * 1024_int64)

    print *, "Time taken for ", iter, " decompressions: ", diff, " seconds"
    print *, "Throughput: ", gb / diff, " GB/s"

    ! Print the output data
    !    do i = 1, size(output_data)
    !        print *, "output_data(", i, ") = ", output_data(i)
    !    end do
end program main