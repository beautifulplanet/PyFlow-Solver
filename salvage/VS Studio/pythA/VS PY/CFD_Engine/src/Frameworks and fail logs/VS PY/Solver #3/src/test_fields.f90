program test_fields
  use parameters
  use fields
  implicit none
  call init_parameters()
  call allocate_fields()
  if (.not. allocated(u) .or. .not. allocated(v) .or. .not. allocated(p)) then
    print *, 'FAIL: fields not allocated.'
    stop 1
  end if
  if (any(u /= 0.0) .or. any(v /= 0.0) .or. any(p /= 0.0)) then
    print *, 'FAIL: fields not zero-initialized.'
    stop 1
  end if
  print *, 'test_fields: PASS'
end program test_fields
