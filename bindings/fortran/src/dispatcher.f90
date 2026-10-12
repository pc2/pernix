module pernix_dispatcher
    use iso_c_binding, only : c_int, c_ptr
    implicit none

    integer(c_int), parameter :: PERNIX_BACKEND_AUTO = 0_c_int
    integer(c_int), parameter :: PERNIX_BACKEND_FALLBACK = 1_c_int
    integer(c_int), parameter :: PERNIX_STATUS_OK = 0_c_int

    interface
        function pernix_dispatcher_create(backend, dispatcher) bind(C, name = "pernix_dispatcher_create") result(status)
            import :: c_int, c_ptr
            integer(c_int), value :: backend
            type(c_ptr), intent(out) :: dispatcher
            integer(c_int) :: status
        end function

        subroutine pernix_dispatcher_destroy(dispatcher) bind(C, name = "pernix_dispatcher_destroy")
            import :: c_ptr
            type(c_ptr), value :: dispatcher
        end subroutine
    end interface
end module pernix_dispatcher
