program test_bc
  use parameters
  use fields
  use boundary_conditions, only: apply_bc
  implicit none
  integer :: i,j
  logical :: ok
  real :: eps
  integer :: fail_count

  eps = 1.0e-7
  call init_parameters()
  call allocate_fields()

  ! Set interior to non-zero values to ensure BC overwrite
  do j=2,ny-1
    do i=2,nx-1
      u(i,j) = 1.0
      v(i,j) = 2.0
    end do
  end do
  call apply_bc()

  ok = .true.
  fail_count = 0

  ! Left/right walls excluding top lid corners (corners take lid velocity)
  do j=1,ny-1
    if (abs(u(1,j)) > eps .or. abs(v(1,j)) > eps) then
      ok = .false.; fail_count = fail_count + 1
      if (fail_count < 10) write(*,*) 'Fail: left wall at j=', j, ' u=',u(1,j),' v=',v(1,j)
    end if
    if (abs(u(nx,j)) > eps .or. abs(v(nx,j)) > eps) then
      ok = .false.; fail_count = fail_count + 1
      if (fail_count < 10) write(*,*) 'Fail: right wall at j=', j, ' u=',u(nx,j),' v=',v(nx,j)
    end if
  end do

  ! Bottom wall
  do i=1,nx
    if (abs(u(i,1)) > eps .or. abs(v(i,1)) > eps) then
      ok = .false.; fail_count = fail_count + 1
      if (fail_count < 10) write(*,*) 'Fail: bottom at i=', i, ' u=',u(i,1),' v=',v(i,1)
    end if
  end do

  ! Top lid (all points including corners should have u=u_lid, v=0)
  do i=1,nx
    if (abs(u(i,ny) - u_lid) > eps) then
      ok = .false.; fail_count = fail_count + 1
      if (fail_count < 10) write(*,*) 'Fail: lid u mismatch at i=', i, ' u=',u(i,ny)
    end if
    if (abs(v(i,ny)) > eps) then
      ok = .false.; fail_count = fail_count + 1
      if (fail_count < 10) write(*,*) 'Fail: lid v mismatch at i=', i, ' v=',v(i,ny)
    end if
  end do

  open(unit=20,file='output/bc_test.dat',status='replace',action='write')
  write(20,*) '# Boundary condition test summary'
  write(20,*) 'nx ny =', nx, ny
  write(20,*) 'u_lid =', u_lid
  write(20,*) 'fail_count =', fail_count
  write(20,*) 'status =', merge('PASS','FAIL',ok)
  close(20)

  if (ok) then
    write(*,*) 'BC test passed.'
  else
    write(*,*) 'BC test FAILED with', fail_count, 'issues.'
    stop 1
  end if
end program test_bc