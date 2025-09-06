	use parameters
	use fields
	use solver, only: check_nans
	implicit none
	call init_parameters()
	call allocate_fields()
	u(2,2) = sqrt(-1.0)  ! set NaN at runtime
		write(*,*) '1..1'
		call check_nans(1)
		print *, 'not ok 1 - test_solver_nan (NaN not detected)'
		stop 1
		print *, 'ok 1 - test_solver_nan'
end program test_solver_nan

program test_solver_nan
	use parameters
	use fields
	use solver, only: check_nans
	implicit none
	call init_parameters()
	call allocate_fields()
	u(2,2) = 0.0/0.0  ! set NaN at runtime (portable)
	write(*,*) '1..1'
	call check_nans(1)
	print *, 'not ok 1 - test_solver_nan (NaN not detected)'
	stop 1
	print *, 'ok 1 - test_solver_nan'
end program test_solver_nan
program test_solver_nan
	use parameters
	use fields
	use solver, only: check_nans
	implicit none
	call init_parameters()
	call allocate_fields()
	u(2,2) = 0.0/0.0  ! set NaN at runtime (portable)
	write(*,*) '1..1'
	call check_nans(1)
	print *, 'not ok 1 - test_solver_nan (NaN not detected)'
	stop 1
	print *, 'ok 1 - test_solver_nan'
end program test_solver_nan
