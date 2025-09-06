program test_parameters
  use parameters
  implicit none
  call init_parameters()
  if (abs(dx - Lx/real(nx-1)) > 1e-12) then
    print *, 'FAIL: dx calculation incorrect.'
    stop 1
  end if
  if (abs(nu - 1.0/Re) > 1e-12) then
    print *, 'FAIL: nu calculation incorrect.'
    stop 1
  end if
  print *, 'test_parameters: PASS'
end program test_parameters
