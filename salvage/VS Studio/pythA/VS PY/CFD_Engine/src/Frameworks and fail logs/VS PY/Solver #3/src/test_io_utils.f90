program test_io_utils
  use parameters
  use fields
  use io_utils, only: write_output
  implicit none
  call init_parameters()
  call allocate_fields()
  u(1,1) = 42.0
  call write_output('output/test_io.dat')
  print *, 'test_io_utils: PASS (check output/test_io.dat for content)'
end program test_io_utils
