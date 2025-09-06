module parameters
  implicit none
  ! Physical parameters
  real, public :: Re = 100.0
  real, public :: rho = 1.0
  real, public :: nu  ! kinematic viscosity = 1/Re

  ! Domain size (Lx = Ly = 1 for cavity)
  real, public :: Lx = 1.0, Ly = 1.0

  ! Grid
  integer, public :: nx = 32, ny = 32
  real, public :: dx, dy, dx2, dy2

  ! Time stepping
  real, public :: dt = 0.001
  integer, public :: max_iter = 5000
  real, public :: tol = 1.0e-6  ! velocity residual stop criterion
  integer, public :: log_interval = 100  ! residual logging frequency

  ! Pressure Poisson parameters
  integer, public :: p_max_iter = 500
  real,    public :: p_tol = 1.0e-5
  real,    public :: p_omega = 1.7  ! SOR relaxation factor (1=Gauss-Seidel, 1.5-1.7 typical)

  ! Lid velocity
  real, public :: u_lid = 1.0

contains
  subroutine init_parameters()
    dx = Lx / real(nx - 1)
    dy = Ly / real(ny - 1)
    dx2 = dx*dx
    dy2 = dy*dy
    nu = 1.0 / Re
  end subroutine init_parameters
end module parameters
