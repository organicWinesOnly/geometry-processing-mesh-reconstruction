#include "poisson_surface_reconstruction.h"
#include "fd_interpolate.h"
#include "fd_grad.h"
#include <igl/copyleft/marching_cubes.h>
#include <algorithm>
#include <iostream>

void poisson_surface_reconstruction(
    const Eigen::MatrixXd & P,
    const Eigen::MatrixXd & N,
    Eigen::MatrixXd & V,
    Eigen::MatrixXi & F)
{
  ////////////////////////////////////////////////////////////////////////////
  // Construct FD grid, CONGRATULATIONS! You get this for free!
  ////////////////////////////////////////////////////////////////////////////
  // number of input points
  const int n = P.rows();
  // Grid dimensions
  int nx, ny, nz;
  // Maximum extent (side length of bounding box) of points
  double max_extent =
    (P.colwise().maxCoeff()-P.colwise().minCoeff()).maxCoeff();
  // padding: number of cells beyond bounding box of input points
  const double pad = 8;
  // choose grid spacing (h) so that shortest side gets 30+2*pad samples
  double h  = max_extent/double(30+2*pad);
  // Place bottom-left-front corner of grid at minimum of points minus padding
  Eigen::RowVector3d corner = P.colwise().minCoeff().array()-pad*h;
  // Grid dimensions should be at least 3 
  nx = std::max((P.col(0).maxCoeff()-P.col(0).minCoeff()+(2.*pad)*h)/h,3.);
  ny = std::max((P.col(1).maxCoeff()-P.col(1).minCoeff()+(2.*pad)*h)/h,3.);
  nz = std::max((P.col(2).maxCoeff()-P.col(2).minCoeff()+(2.*pad)*h)/h,3.);
  // Compute positions of grid nodes
  Eigen::MatrixXd x(nx*ny*nz, 3);
  for(int i = 0; i < nx; i++) 
  {
    for(int j = 0; j < ny; j++)
    {
      for(int k = 0; k < nz; k++)
      {
         // Convert subscript to index
         const auto ind = i + nx*(j + k * ny);
         x.row(ind) = corner + h*Eigen::RowVector3d(i,j,k);
      }
    }
  }

  ////////////////////////////////////////////////////////////////////////////
  // Add your code here
  ////////////////////////////////////////////////////////////////////////////
  int m1, m2, m3;  // row dimensions of vx, vy nd vz respectively
  m1 = (nx-1) * ny * nz;
  m2 = nx * (ny-1) * nz;
  m3 = nx * ny * (nz-1);
  Eigen::VectorXd vx(m1); 
  Eigen::VectorXd vy(m2); 
  Eigen::VectorXd vz(m3); 
  Eigen::VectorXd v(m1+m2+m3); 

  Eigen::SparseMatrix<double> weight_x;  // interpolation weights 
  Eigen::SparseMatrix<double> weight_y;
  Eigen::SparseMatrix<double> weight_z;
  Eigen::SparseMatrix<double> weight;

  // shift the location of the bottom left corner of the primary grid to match
  // the x-, y-, z- staggered grids respectively
  Eigen::RowVector3d corner_x = corner;
  corner_x(0) = corner(0) + h/2;
  Eigen::RowVector3d corner_y = corner;
  corner_y(1) = corner(1) + h/2;
  Eigen::RowVector3d corner_z = corner;
  corner_z(2) = corner(2) + h/2;
  assert(&corner_x != &corner_y);

  fd_interpolate(nx-1, ny, nz, h, corner_x, P, weight_x);
  fd_interpolate(nx, ny-1, nz, h, corner_y, P, weight_y);
  fd_interpolate(nx, ny, nz-1, h, corner_z, P, weight_z);
  fd_interpolate(nx, ny, nz, h, corner, P, weight);
  assert(P.isApprox(weight*x, 1e-8));
  
  vx = weight_x.transpose() * N.col(0);
  vy = weight_y.transpose() * N.col(1);
  vz = weight_z.transpose() * N.col(2);
  v.head(m1) = vx;
  v.segment(m1, m2) = vy;
  v.tail(m3) = vz;

  Eigen::SparseMatrix<double> G, L;  // Gradient and Laplacian
  fd_grad(nx, ny, nz, h, G);
  L = Eigen::SparseMatrix<double>(G.transpose()) * G; 

  ///////////////////////////////////////////////////////////////////////////
  // Sparse solver
  ///////////////////////////////////////////////////////////////////////////
  Eigen::BiCGSTAB<Eigen::SparseMatrix<double> > solver;
  solver.compute(L);

  Eigen::VectorXd g(nx * ny * nz); 
  g = solver.solve(Eigen::SparseMatrix<double>(G.transpose()) * v);
  
  ////////////////////////////////////////////////////////////////////////////
  // Run black box algorithm to compute mesh from implicit function: this
  // function always extracts g=0, so "pre-shift" your g values by -sigma
  ////////////////////////////////////////////////////////////////////////////
  double sigma;
  sigma = (1/n) * Eigen::RowVectorXd::Ones(n).eval() * (weight * g); 

  g = g.array() - sigma;
  igl::copyleft::marching_cubes(g, x, nx, ny, nz, V, F);
}
