	use parameters
	use fields
	implicit none
	call init_parameters()
	call allocate_fields()
		write(*,*) '1..1'
		if (.not. allocated(u) .or. .not. allocated(v) .or. .not. allocated(p)) then
			print *, 'not ok 1 - test_fields (not allocated)'
			stop 1
		end if
		if (any(u /= 0.0) .or. any(v /= 0.0) .or. any(p /= 0.0)) then
			print *, 'not ok 1 - test_fields (not zero-initialized)'
			stop 1
		end if
		print *, 'ok 1 - test_fields'
end program test_fields

program test_fields
	use parameters
	use fields
	implicit none
	call init_parameters()
	call allocate_fields()
	write(*,*) '1..1'
	if (.not. allocated(u) .or. .not. allocated(v) .or. .not. allocated(p)) then
		print *, 'not ok 1 - test_fields (not allocated)'
		stop 1
	end if
	if (any(u /= 0.0) .or. any(v /= 0.0) .or. any(p /= 0.0)) then
		print *, 'not ok 1 - test_fields (not zero-initialized)'
		stop 1
	end if
	print *, 'ok 1 - test_fields'
end program test_fields
program test_fields
	use parameters
	use fields
	implicit none
	call init_parameters()
	call allocate_fields()
	write(*,*) '1..1'
	if (.not. allocated(u) .or. .not. allocated(v) .or. .not. allocated(p)) then
		print *, 'not ok 1 - test_fields (not allocated)'
		stop 1
	end if
	if (any(u /= 0.0) .or. any(v /= 0.0) .or. any(p /= 0.0)) then
		print *, 'not ok 1 - test_fields (not zero-initialized)'
		stop 1
	end if
	print *, 'ok 1 - test_fields'
end program test_fields
