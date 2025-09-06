	use parameters
	implicit none
	call init_parameters()
		write(*,*) '1..1'
		if (abs(dx - Lx/real(nx-1)) > 1e-12) then
			print *, 'not ok 1 - test_parameters (dx)'
			stop 1
		end if
		if (abs(nu - 1.0/Re) > 1e-12) then
			print *, 'not ok 1 - test_parameters (nu)'
			stop 1
		end if
		print *, 'ok 1 - test_parameters'
end program test_parameters

program test_parameters
	use parameters
	implicit none
	call init_parameters()
	write(*,*) '1..1'
	if (abs(dx - Lx/real(nx-1)) > 1e-12) then
		print *, 'not ok 1 - test_parameters (dx)'
		stop 1
	end if
	if (abs(nu - 1.0/Re) > 1e-12) then
		print *, 'not ok 1 - test_parameters (nu)'
		stop 1
	end if
	print *, 'ok 1 - test_parameters'
end program test_parameters
program test_parameters
	use parameters
	implicit none
	call init_parameters()
	write(*,*) '1..1'
	if (abs(dx - Lx/real(nx-1)) > 1e-12) then
		print *, 'not ok 1 - test_parameters (dx)'
		stop 1
	end if
	if (abs(nu - 1.0/Re) > 1e-12) then
		print *, 'not ok 1 - test_parameters (nu)'
		stop 1
	end if
	print *, 'ok 1 - test_parameters'
end program test_parameters
