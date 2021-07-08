#include <iostream>
#include "fd_grad.h"
#include "fd_partial_derivative.h"
// TODO write some test

void fd_grad(
  const int nx,
  const int ny,
  const int nz,
  const double h,
  Eigen::SparseMatrix<double> & G)
{
  int m1 = (nx-1) * ny * nz;
  int m2 = nx * (ny-1) * nz;
  int m3 = nx * ny * (nz-1);
  int m = m1 + m2 + m3;
  int n = nx * ny * nz;

  Eigen::SparseMatrix<double> Dx;
  Eigen::SparseMatrix<double> Dy;
  Eigen::SparseMatrix<double> Dz;
  std::cout << "partial" << std::endl;
  fd_partial_derivative(nx, ny, nz, h, 0, Dx);  
  std::cout << "x partial" << std::endl;
  fd_partial_derivative(nx, ny, nz, h, 1, Dy);  
  std::cout << "x partial" << std::endl;
  fd_partial_derivative(nx, ny, nz, h, 2, Dz); 
  std::cout << "x partial" << std::endl;

  Eigen::SparseMatrix<double> itermed(n, m);
  G.resize(m, n);
  itermed.leftCols(m1) = Dx.transpose();
  itermed.middleCols(m1, m2) = Dy.transpose();
  itermed.rightCols(m3) = Dz.transpose();
  G = itermed.transpose();
  assert(G.block(0,0,m1,n).isApprox(Dx, 1e-8));
  assert(G.block(m1,0,m2,n).isApprox(Dy, 1e-8));
  assert(G.block(m1+m2,0,m3,n).isApprox(Dz, 1e-8));

}
