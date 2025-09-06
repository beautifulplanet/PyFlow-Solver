program hpf_cfd
  use parameters
  use fields
  use boundary_conditions, only: apply_bc
  use solver, only: step
  use io_utils, only: write_output
  implicit none
  integer :: iter
  real :: res_u, res_v
  integer :: unit_res

  call init_parameters()
  call allocate_fields()
  call apply_bc()

  ! Residual log file
  open(newunit=unit_res, file='output/residuals.dat', status='replace', action='write')
  write(unit_res,'(A)') '# iter  res_u  res_v'

  do iter = 1, max_iter
     call step(iter, res_u, res_v)
     if (mod(iter,log_interval)==0) then
        write(*,'(A,I6,2F12.5)') 'Iter:', iter, res_u, res_v
        write(unit_res,'(I8,2E16.6)') iter, res_u, res_v
     end if
     if (max(res_u,res_v) < tol) then
        write(*,'(A,I6,2F12.5)') 'Converged at:', iter, res_u, res_v
        write(unit_res,'(I8,2E16.6)') iter, res_u, res_v
        exit
     end if
  end do

  call write_output('output/results.dat')
  write(*,*) 'Simulation complete.'
  close(unit_res)
end program hpf_cfd
