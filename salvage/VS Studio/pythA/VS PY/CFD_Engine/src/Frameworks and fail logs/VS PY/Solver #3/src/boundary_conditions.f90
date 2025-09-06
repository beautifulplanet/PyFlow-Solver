module boundary_conditions
  use parameters, only: nx, ny, u_lid
  use fields, only: u, v
  implicit none
contains
  subroutine apply_bc()
    integer :: i, j
    ! Left & right walls (u=v=0)
    do j=1,ny
      u(1,j) = 0.0; v(1,j) = 0.0
      u(nx,j)= 0.0; v(nx,j)= 0.0
    end do
    ! Bottom wall
    do i=1,nx
      u(i,1) = 0.0; v(i,1) = 0.0
    end do
    ! Top lid: u = u_lid, v = 0
    do i=1,nx
      u(i,ny) = u_lid
      v(i,ny) = 0.0
    end do
  end subroutine apply_bc
end module boundary_conditions
