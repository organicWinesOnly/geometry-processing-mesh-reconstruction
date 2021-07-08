#include "fd_partial_derivative.h"
#include <vector>

void partial_x_direction(
    const int nx, 
    const int ny, 
    const int nz, 
    const int m, 
    const double h,
    Eigen::SparseMatrix<double> &D);

void partial_y_direction(
    const int nx, 
    const int ny, 
    const int nz, 
    const int m, 
    const double h,
    Eigen::SparseMatrix<double> &D);

void partial_z_direction(
    const int nx, 
    const int ny, 
    const int nz, 
    const int m, 
    const double h,
    Eigen::SparseMatrix<double> &D);

typedef Eigen::Triplet<double> T;
void fd_partial_derivative(
  const int nx,
  const int ny,
  const int nz,
  const double h,
  const int dir,
  Eigen::SparseMatrix<double> & D)
{
  int n = nx * ny * nz;
  int m;

  switch (dir)
  {
    case 0 :
      m = (nx-1) * ny * nz; 
      partial_x_direction(nx, ny, nz, m, h, D);
      break;
    case 1 :
      m = nx * (ny-1) * nz; 
      partial_y_direction(nx, ny, nz, m, h, D);
      break;
    default :
      m = nx * ny * (nz-1);
      partial_z_direction(nx, ny, nz, m, h, D);
  }
}


void partial_x_direction(
    const int nx, 
    const int ny, 
    const int nz, 
    const int m, 
    const double h,
    Eigen::SparseMatrix<double> &D)
{

  int n = nx * ny * nz;
  std::vector<T> non_zero_coeff(2 * m);
  for (int l = 0; l < (nx - 1); l++)
  {
    for (int j = 0; j < ny; j++)
    {
      for (int k = 0; k < nz; k++)
      {
        const auto ind0 = l + (nx-1) * (j + k * ny);
	// ind_primary_<> lives on the primary grid not the staggered grid
        const auto ind_primary_0 = l + nx * (j + k * ny);
        const auto ind_primary_1 = (l+1) + nx * (j + k * ny);
	non_zero_coeff.push_back(T(ind0, ind_primary_0, -1/h));
	non_zero_coeff.push_back(T(ind0, ind_primary_1, 1/h));
      }
    }
  }

  D.resize(m, n);
  D.setFromTriplets(non_zero_coeff.begin(), non_zero_coeff.end());
}


void partial_y_direction(
    const int nx, 
    const int ny, 
    const int nz, 
    const int m, 
    const double h,
    Eigen::SparseMatrix<double> &D)
{

  int n = nx * ny * nz;
  std::vector<T> non_zero_coeff(2 * m);
  // for Dx
  for (int i = 0; i < nx; i++)
  {
    for (int l = 0; l < (ny-1); l++)
    {
      for (int k = 0; k < nz; k++)
      {
        const auto ind0 = i + nx * (l + k * (ny-1));
	// ind_primary_<> live on the primary grid not the staggered grid
        const auto ind_primary_0 = i + nx * (l + k * ny);
        const auto ind_primary_1 = i + nx * (l+1 + k * ny);
	non_zero_coeff.push_back(T(ind0, ind_primary_0, -1/h));
	non_zero_coeff.push_back(T(ind0, ind_primary_1, 1/h));
      }
    }
  }

  D.resize(m, n);
  D.setFromTriplets(non_zero_coeff.begin(), non_zero_coeff.end());
}

void partial_z_direction(
    const int nx, 
    const int ny, 
    const int nz, 
    const int m, 
    const double h,
    Eigen::SparseMatrix<double> &D)
{

  int n = nx * ny * nz;
  std::vector<T> non_zero_coeff(2 * m);
  for (int i = 0; i < nx; i++)
  {
    for (int j = 0; j < ny; j++)
    {
      for (int l = 0; l < (nz-1); l++)
      {
        const auto ind0 = i + nx * (j + l * ny);
        const auto ind_primary_0 = i + nx * (j + l * ny);
        const auto ind_primary_1 = i + nx * (j + (l+1) * ny);
	non_zero_coeff.push_back(T(ind0, ind_primary_0, -1/h));
	non_zero_coeff.push_back(T(ind0, ind_primary_1, 1/h));
      }
    }
  }

  D.resize(m, n);
  D.setFromTriplets(non_zero_coeff.begin(), non_zero_coeff.end());
}
