	use parameters
	use fields
	use io_utils, only: write_output
	implicit none
	call init_parameters()
	call allocate_fields()
	u(1,1) = 42.0
	call write_output('output/test_io.dat')
		write(*,*) '1..1'
		print *, 'ok 1 - test_io_utils'
end program test_io_utils

program test_io_utils
	use parameters
	use fields
	use io_utils, only: write_output
	implicit none
	call init_parameters()
	call allocate_fields()
	u(1,1) = 42.0
	call write_output('output/test_io.dat')
	write(*,*) '1..1'
	print *, 'ok 1 - test_io_utils'
end program test_io_utils
program test_io_utils
	use parameters
	use fields
	use io_utils, only: write_output
	implicit none
	call init_parameters()
	call allocate_fields()
	u(1,1) = 42.0
	call write_output('output/test_io.dat')
	write(*,*) '1..1'
	print *, 'ok 1 - test_io_utils'
end program test_io_utils
