module solver
  use parameters
  use fields
  use boundary_conditions, only: apply_bc
  use, intrinsic :: ieee_arithmetic
  implicit none
contains
  subroutine step(iter, res_u, res_v)
    integer, intent(in) :: iter
    real, intent(out) :: res_u, res_v

    call store_old()
    call momentum_predictor()
    call build_pressure_rhs()
    call pressure_poisson()
    call velocity_correction()
    call apply_bc()
    call compute_residuals(res_u, res_v)
    call check_nans(iter)
  end subroutine step

  subroutine store_old()
    u_old = u
    v_old = v
  end subroutine store_old

  subroutine momentum_predictor()
    ! Explicit first-order upwind convection, central diffusion
    integer :: i,j
    real :: du_dx, du_dy, dv_dx, dv_dy
    real :: d2u_dx2, d2u_dy2, d2v_dx2, d2v_dy2
    do j=2,ny-1
      do i=2,nx-1
        ! Upwind for u convection term components
        if (u(i,j) > 0.0) then
          du_dx = (u(i,j) - u(i-1,j)) / dx
        else
          du_dx = (u(i+1,j) - u(i,j)) / dx
        end if
        if (v(i,j) > 0.0) then
          du_dy = (u(i,j) - u(i,j-1)) / dy
        else
          du_dy = (u(i,j+1) - u(i,j)) / dy
        end if
        if (u(i,j) > 0.0) then
          dv_dx = (v(i,j) - v(i-1,j)) / dx
        else
          dv_dx = (v(i+1,j) - v(i,j)) / dx
        end if
        if (v(i,j) > 0.0) then
          dv_dy = (v(i,j) - v(i,j-1)) / dy
        else
          dv_dy = (v(i,j+1) - v(i,j)) / dy
        end if
        d2u_dx2 = (u(i+1,j) - 2.0*u(i,j) + u(i-1,j)) / dx2
        d2u_dy2 = (u(i,j+1) - 2.0*u(i,j) + u(i,j-1)) / dy2
        d2v_dx2 = (v(i+1,j) - 2.0*v(i,j) + v(i-1,j)) / dx2
        d2v_dy2 = (v(i,j+1) - 2.0*v(i,j) + v(i,j-1)) / dy2

        u_star(i,j) = u(i,j) + dt * ( - ( u(i,j)*du_dx + v(i,j)*du_dy ) + nu*(d2u_dx2 + d2u_dy2) )
        v_star(i,j) = v(i,j) + dt * ( - ( u(i,j)*dv_dx + v(i,j)*dv_dy ) + nu*(d2v_dx2 + d2v_dy2) )
      end do
    end do
    call apply_bc_star()
  end subroutine momentum_predictor

  subroutine apply_bc_star()
    integer :: i,j
    ! Mirror boundary treatment for predicted velocities then apply physical BC
    do j=1,ny
      u_star(1,j)=0.0; v_star(1,j)=0.0
      u_star(nx,j)=0.0; v_star(nx,j)=0.0
    end do
    do i=1,nx
      u_star(i,1)=0.0; v_star(i,1)=0.0
      u_star(i,ny)=u_lid; v_star(i,ny)=0.0
    end do
  end subroutine apply_bc_star

  subroutine build_pressure_rhs()
    integer :: i,j
    real :: du_dx, dv_dy
    rhs_p = 0.0
    do j=2,ny-1
      do i=2,nx-1
        du_dx = (u_star(i+1,j) - u_star(i-1,j)) / (2.0*dx)
        dv_dy = (v_star(i,j+1) - v_star(i,j-1)) / (2.0*dy)
        rhs_p(i,j) = (rho/dt) * (du_dx + dv_dy)
      end do
    end do
  end subroutine build_pressure_rhs

  subroutine pressure_poisson()
    integer :: it, i, j
    real :: err, p_new, omega
    omega = p_omega
    do it=1,p_max_iter
      err = 0.0
      do j=2,ny-1
        do i=2,nx-1
          p_new = ((p(i+1,j)+p(i-1,j))*dy2 + (p(i,j+1)+p(i,j-1))*dx2 - rhs_p(i,j)*dx2*dy2) / (2.0*(dx2+dy2))
          ! SOR update
          p_new = (1.0-omega)*p(i,j) + omega*p_new
          err = err + (p_new - p(i,j))**2
          p(i,j) = p_new
        end do
      end do
      call pressure_bc()
      if (err/(nx*ny) < p_tol) exit
    end do
  end subroutine pressure_poisson

  subroutine pressure_bc()
    integer :: i,j
    ! Neumann (zero normal gradient) on all walls; fix one reference point to zero
    do j=1,ny
      p(1,j) = p(2,j)
      p(nx,j)= p(nx-1,j)
    end do
    do i=1,nx
      p(i,1) = p(i,2)
      p(i,ny)= p(i,ny-1)
    end do
    p(1,1) = 0.0  ! reference pressure
  end subroutine pressure_bc

  subroutine velocity_correction()
    integer :: i,j
    do j=2,ny-1
      do i=2,nx-1
        u(i,j) = u_star(i,j) - dt/rho * (p(i+1,j) - p(i-1,j)) / (2.0*dx)
        v(i,j) = v_star(i,j) - dt/rho * (p(i,j+1) - p(i,j-1)) / (2.0*dy)
      end do
    end do
  end subroutine velocity_correction

  subroutine compute_residuals(res_u, res_v)
    real, intent(out) :: res_u, res_v
    real :: sum_u, sum_v
    sum_u = sum( (u - u_old)**2 )
    sum_v = sum( (v - v_old)**2 )
    res_u = sqrt( sum_u / real(nx*ny) )
    res_v = sqrt( sum_v / real(nx*ny) )
  end subroutine compute_residuals

  subroutine check_nans(iter)
    integer, intent(in) :: iter
    logical :: has_nan
    has_nan = any(ieee_is_nan(u)) .or. any(ieee_is_nan(v)) .or. any(ieee_is_nan(p))
    if (has_nan) then
      write(*,*) 'NaN detected at iteration', iter, ' aborting.'
      stop 1
    end if
  end subroutine check_nans
end module solver
