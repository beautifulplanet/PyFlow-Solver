program test_solver_nan
  use parameters
  use fields
  use solver, only: check_nans
  implicit none
  call init_parameters()
  call allocate_fields()
  u(2,2) = 0.0/0.0  ! set NaN
  call check_nans(1)
  print *, 'FAIL: NaN not detected.'
  stop 1
end program test_solver_nan
