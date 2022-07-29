module adjoint_write_files

    implicit none
    public :: write_files_derivative
    public :: write_files_delta

    private
    
contains

subroutine write_p_values

    use stella_geometry, only: geo_surf
    use millerlocal, only: np0
    implicit none 

    integer :: ip 
    real, dimension (:), allocatable :: p_out 

    allocate (p_out(np0))

    p_out(1) = geo_surf%rhoc
    p_out(2) = geo_surf%rmaj 
    p_out(3) = geo_surf%rgeo
    p_out(4) = geo_surf%shift
    p_out(5) = geo_surf%kappa
    p_out(6) = geo_surf%kapprim
    p_out(7) = geo_surf%qinp
    p_out(8) = geo_surf%shat
    p_out (9) = geo_surf%tri
    p_out (10) = geo_surf%triprim
    p_out (11) = geo_surf%betaprim

    open(15, file="adjoint_files/adjoint_p_values.dat", status="unknown",action="write",position="replace")
    do ip = 1, np0
        write(15,*) p_out(ip)
    end do
    close(15)

end subroutine write_p_values

subroutine write_files_omega

    use adjoint_field_arrays, only: omega_g

    implicit none

    open(13, file="adjoint_files/adjoint_omega.dat", status="unknown",action="write",position="replace")
    write(13,*) real(omega_g), aimag(omega_g)
    close(13)

end subroutine write_files_omega

subroutine write_files_delta

    use millerlocal, only: del, np0

    implicit none

    integer :: ip 

    open(14, file="adjoint_files/adjoint_delta.dat", status="unknown",action="write",position="replace")
    do ip = 1, np0
        write(14,*) del
    end do
    close(14)

    call write_files_omega
    call write_p_values

end subroutine write_files_delta

subroutine write_files_derivative (adjoint_var, derivative,new_file)

    use adjoint_field_arrays, only: omega_g
    use stella_geometry, only: geo_surf

    implicit none
    
    integer, intent (in) :: adjoint_var
    complex, dimension (:,:), intent (in) :: derivative
    logical, intent (in) :: new_file
    
    if(new_file) then
       open(12, file="adjoint_files/adjoint_derivatives.dat", status="unknown",action="write",position="replace")
       write(12,*) real(derivative(1,1)), aimag(derivative(1,1))
       close(12)
    else
       open(12, file="adjoint_files/adjoint_derivatives.dat", status="unknown",action="write",position="append")
       write(12,*) real(derivative(1,1)), aimag(derivative(1,1))
       close(12)
    end if

end subroutine write_files_derivative


end module adjoint_write_files
