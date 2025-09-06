module io_utils
  use parameters, only: nx, ny
  use fields, only: u, v, p
  implicit none
contains
  subroutine write_output(filename)
    character(len=*), intent(in) :: filename
    integer :: i, j
    open(unit=10, file=filename, status='replace', action='write')
    write(10,*) '# i j u v p'
    do j=1,ny
      do i=1,nx
        write(10,'(2I6,3F15.6)') i, j, u(i,j), v(i,j), p(i,j)
      end do
    end do
    close(10)
  end subroutine write_output
end module io_utils
