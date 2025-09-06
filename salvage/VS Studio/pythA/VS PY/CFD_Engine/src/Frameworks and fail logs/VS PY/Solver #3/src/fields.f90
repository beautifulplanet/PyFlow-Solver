module fields
  use parameters, only: nx, ny
  implicit none
  ! Primary fields (collocated)
  real, allocatable, public :: u(:,:), v(:,:), p(:,:)
  ! Work / auxiliary arrays
  real, allocatable, public :: u_star(:,:), v_star(:,:)
  real, allocatable, public :: u_old(:,:), v_old(:,:)
  real, allocatable, public :: rhs_p(:,:)
contains
  subroutine allocate_fields()
    allocate(u(nx,ny), v(nx,ny), p(nx,ny))
    allocate(u_star(nx,ny), v_star(nx,ny))
    allocate(u_old(nx,ny), v_old(nx,ny))
    allocate(rhs_p(nx,ny))
    u = 0.0; v = 0.0; p = 0.0
    u_star = 0.0; v_star = 0.0
    u_old = 0.0; v_old = 0.0
    rhs_p = 0.0
  end subroutine allocate_fields
end module fields
