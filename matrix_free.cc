// steady state linear heat conduction equation
// jcr, 2013

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/numbers.h>

#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/numerics/data_out.h>
//### include <deal.II/numerics/vectors.h>
//### include <deal.II/numerics/matrices.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <fstream>
#include <iostream>

using namespace dealii;

void compute_thermal_conductivity(const std::vector<double> &T,
                                  std::vector<double> &values,
                                  std::vector<double> &derivatives);

//------------------------------------------------------
template <int dim>
void compute_nonlinear_residual(DoFHandler<dim> const &  dof_handler, 
                                ConstraintMatrix const & constraints,
                                Vector<double> const &   solution,
                                Vector<double> &         nonlinear_residual);
//------------------------------------------------------
template <int dim>
class ActionOfJacobianOnVector {
public:
  ActionOfJacobianOnVector(DoFHandler<dim> const &  dof_handler,
                           ConstraintMatrix const & constraints,
                           Vector<double> const &   solution, 
                           Vector<double> const &   minus_unperturbed_residual,
                           Vector<double> &         perturbed_residual)
  : _dof_handler(dof_handler),
    _constraints(constraints),
    _solution(solution),
    _minus_unperturbed_residual(minus_unperturbed_residual),
    _perturbed_residual(perturbed_residual)
  { }
  void vmult(Vector<double> &u, const Vector<double> &v) const;
  void Tvmult(Vector<double> &u, const Vector<double> &v) const {
    Assert(false, ExcNotImplemented());
  }
  void vmult_add(Vector<double> &u, const Vector<double> &v) const {
    Assert(false, ExcNotImplemented());
  }
  void Tvmult_add(Vector<double> &u, const Vector<double> &v) const {
    Assert(false, ExcNotImplemented());
  }
private:
  DoFHandler<dim> const &  _dof_handler;
  ConstraintMatrix const & _constraints;
  Vector<double> const &   _solution;
  Vector<double> const &   _minus_unperturbed_residual;
  Vector<double> &         _perturbed_residual;
};

template <int dim>
void ActionOfJacobianOnVector<dim>::vmult(Vector<double> &u, const Vector<double> &v) const {
  // compute epsilon
  double epsilon = 0.0;
  size_t const n = _solution.size();
  double const b = sqrt(1.0e-15);
  double const v_l2_norm = v.l2_norm();
  for (size_t i = 0; i < n; ++i) {
    epsilon += b * (1.0 + numbers::NumberTraits<double>::abs(_solution[i]));
  } // end for i
  epsilon /= (static_cast<double>(n) * v_l2_norm);
  epsilon=1234.0;
  std::cout << "here " << std::endl;
  std::cout << "b=" << b << "  n=" << n << "  v_l2_norm=" << v_l2_norm << "  epsilon=" << epsilon << std::endl ;

  // perturb u
  u = _solution;
  u.add(epsilon, v);
  
  std::cout << "x=" ;
  _solution.print(std::cout, 15);
  std::cout << std::endl;
  std::cout << "x+epsilon.v=";
  u.print(std::cout, 15);
  std::cout << std::endl;

  // compute perturbed residual
  compute_nonlinear_residual<dim>(_dof_handler, _constraints, u, _perturbed_residual);

  std::cout << "F(x+epsilon v)=";
  _perturbed_residual.print(std::cout, 15);
  std::cout << std::endl;
  std::cout << "-F(x)=";
  _minus_unperturbed_residual.print(std::cout, 15);
  std::cout << std::endl;

  // compute FD
  u = _perturbed_residual;
  u += _minus_unperturbed_residual;
  u /= epsilon;
}

//------------------------------------------------------
//------------------------------------------------------

template <int dim>
class SS_HC 
{
  public:
    SS_HC ();
    void run ();
    
  private:
    void make_grid ();
    void setup_system();
    void assemble_matrix (unsigned int option); // pass ref?
    void compute_residual ();
    std::pair<unsigned int, double> linear_solve (Vector<double>   &newton_update); // pass name?
    void output_results () const;
    
    Triangulation<dim>   triangulation;
    FE_Q<dim>            fe;
    DoFHandler<dim>      dof_handler;
    ConstraintMatrix     constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double>       solution;
    Vector<double>       newton_update;
    Vector<double>       nonlinear_residual;
    Vector<double>       minus_unperturbed_nonlinear_residual;

};


//------------------------------------------------------

template <int dim>
class VolumetricTerm : public Function<dim> 
{
  public:
    VolumetricTerm () : Function<dim>() {}
    
    virtual double value (const Point<dim>   &p,
        const unsigned int  component = 0) const;
};

//------------------------------------------------------

template <int dim>
class BoundaryValues : public Function<dim> 
{
  public:
    BoundaryValues () : Function<dim>() {}
    
    virtual double value (const Point<dim>   &p,
        const unsigned int  component = 0) const;
};

//------------------------------------------------------

template <int dim>
double VolumetricTerm<dim>::value (const Point<dim> &p,
          const unsigned int /*component*/) const 
{
  double return_value = 0;
  for (unsigned int i=0; i<dim; ++i)
    return_value += 4*std::pow(p(i), 4);
				 // For this example, we choose as right hand
				 // side function to function $4(x^4+y^4)$ in
				 // 2D, or $4(x^4+y^4+z^4)$ in 3D.
  // return return_value;

  return_value = 10.0;
//  std::cout << "q = " << return_value << std::endl ;
  return return_value;
}

//------------------------------------------------------

template <int dim>
double BoundaryValues<dim>::value (const Point<dim> &p,
           const unsigned int /*component*/) const 
{
				 // As boundary values, we choose x*x+y*y in
				 // 2D, and x*x+y*y+z*z in 3D. This happens to
				 // be equal to the square of the vector from
				 // the origin to the point at which we would
				 // like to evaluate the function,
				 // irrespective of the dimension.
//  return p.square();
  return 5.0;
}

//------------------------------------------------------

template <int dim>
SS_HC<dim>::SS_HC ()
    :
    fe (1),
    dof_handler (triangulation)
{}

//------------------------------------------------------

void compute_thermal_conductivity(const std::vector<double> &T,
                                  std::vector<double> &values,
                                  std::vector<double> &derivatives)
{
  const unsigned int n_points = T.size();
  Assert (values.size()      == n_points, ExcDimensionMismatch (values.size(), n_points) );
  const double k0=1.0;
  const double k1=0.0;
    
  for (unsigned int point = 0; point < n_points; ++point){
     values[point]      = k0 + k1*T[point];
     derivatives[point] = k1;
  }
}

//------------------------------------------------------

template <int dim>
void SS_HC<dim>::make_grid ()
{
/*
  const Point<dim> bottom_left = Point<dim>();
  const Point<dim> upper_right = (dim == 2 ? Point<dim> (length,length)
                                           : Point<dim> (length,length,length) 
                                 );
  GridGenerator::hyper_rectangle(triangulation,bottom_left,upper_right); 
*/
  GridGenerator::hyper_cube (triangulation, -1, 1);
  triangulation.refine_global (2);
  
  std::cout << "   Number of active cells: "
            << triangulation.n_active_cells()
            << std::endl
            << "   Total number of cells: "
            << triangulation.n_cells()
            << std::endl;
}

//------------------------------------------------------

template <int dim>
void SS_HC<dim>::setup_system ()
{
  dof_handler.distribute_dofs (fe);

  std::cout << "   Number of degrees of freedom: "
            << dof_handler.n_dofs()
            << std::endl;

  constraints.clear ();
  DoFTools::make_hanging_node_constraints (dof_handler,constraints);
  std::map<unsigned int,double> boundary_values;
  std::vector<bool> mask (1, true);
  VectorTools::interpolate_boundary_values (dof_handler,
                                            0,  //jcr check this
                                            ZeroFunction<dim>(),
                                            constraints,
                                            mask);
  constraints.close(); 

  CompressedSparsityPattern c_sparsity(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern (dof_handler, c_sparsity, constraints, false);
  sparsity_pattern.copy_from(c_sparsity);
  
  system_matrix.reinit (sparsity_pattern);
  
  solution.reinit (dof_handler.n_dofs());
  newton_update.reinit (dof_handler.n_dofs());
  nonlinear_residual.reinit (dof_handler.n_dofs());
  minus_unperturbed_nonlinear_residual.reinit (dof_handler.n_dofs());
}

//------------------------------------------------------

template <int dim>
void compute_nonlinear_residual(DoFHandler<dim> const &  dof_handler, 
                                ConstraintMatrix const & constraints,
                                Vector<double> const &   solution,
                                Vector<double> &         nonlinear_residual) {
  QGauss<dim>  quadrature_formula(2);

  std::cout << "in compute_nonlinear_residual, solution passed =\n     " ;
  solution.print(std::cout, 15);
  std::cout << std::endl;

  // reset to 0
  nonlinear_residual = 0.0;
  
  const VolumetricTerm<dim> volumetric_term;

  FEValues<dim> fe_values(dof_handler.get_fe(), quadrature_formula, 
                          update_values | update_gradients | update_quadrature_points | update_JxW_values);
  const unsigned int dofs_per_cell = (dof_handler.get_fe()).dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  Vector<double> local_f(dofs_per_cell);

  std::vector<double> local_solution_values(n_q_points);
  std::vector<Tensor<1, dim> > local_solution_gradients(n_q_points);

  std::vector<double> conductivity_values(n_q_points);
  std::vector<double> conductivity_derivatives(n_q_points);
  
  std::vector<unsigned int> local_dof_indices (dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
  
  for (; cell!=endc; ++cell) {
    fe_values.reinit (cell);
    local_f = 0.0;
    fe_values.get_function_values   (solution,local_solution_values   );
    fe_values.get_function_gradients(solution,local_solution_gradients);
    compute_thermal_conductivity(local_solution_values,conductivity_values,conductivity_derivatives);
    std::cout << "local solution[q] = ";
    for (unsigned int q_point=0; q_point<n_q_points; ++q_point){
      std::cout << local_solution_values[q_point] << " ";
      std::cout << local_solution_gradients[q_point] << " ";
    }
    std::cout << std::endl;

    for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
      for (unsigned int i=0; i<dofs_per_cell; ++i)
           local_f(i) +=  (fe_values.shape_grad (i, q_point)  *
                           conductivity_values[q_point]       *
                           local_solution_gradients[q_point] 
                           -                            
                           fe_values.shape_value (i, q_point) *
                           volumetric_term.value (fe_values.quadrature_point (q_point)) 
                          )
                          *fe_values.JxW (q_point);
      
    cell->get_dof_indices(local_dof_indices);
    std::cout << "local_f = ";
    for (unsigned int i=0; i<dofs_per_cell; ++i)
      std::cout << local_f(i) << " ";
    std::cout << std::endl;
    constraints.distribute_local_to_global(local_f, local_dof_indices, nonlinear_residual);
  } // end for
  std::cout << "in compute_nonlinear_residual, F=\n    ";
  nonlinear_residual.print(std::cout, 15);
  std::cout << std::endl;
}

template <int dim>
void SS_HC<dim>::compute_residual ()
// -div(k.gradT) = q
// thus
// int( gradbi . k graT -qbi) = 0 + bc
{  
  compute_nonlinear_residual<dim>(dof_handler, constraints, solution, nonlinear_residual);
/*
  QGauss<dim>  quadrature_formula(2);

  // reset to 0
  nonlinear_residual = 0.0;
  
  const VolumetricTerm<dim> volumetric_term;

  FEValues<dim> fe_values (fe, quadrature_formula, 
                           update_values   | update_gradients |
                           update_quadrature_points | update_JxW_values);

  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();

  Vector<double>       local_f (dofs_per_cell);

  std::vector<double>          local_solution_values(n_q_points);
  std::vector<Tensor<1, dim> > local_solution_gradients(n_q_points);

  std::vector<double> conductivity_values(n_q_points);
  std::vector<double> conductivity_derivatives(n_q_points);
  
  std::vector<unsigned int> local_dof_indices (dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
  
  for (; cell!=endc; ++cell)
    {
    fe_values.reinit (cell);
    local_f = 0;
    fe_values.get_function_values   (solution,local_solution_values   );
    fe_values.get_function_gradients(solution,local_solution_gradients);
    compute_thermal_conductivity(local_solution_values,conductivity_values,conductivity_derivatives);

    for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
      for (unsigned int i=0; i<dofs_per_cell; ++i)
           local_f(i) +=  (fe_values.shape_grad (i, q_point)  *
                           conductivity_values[q_point]       *
                           local_solution_gradients[q_point] 
                           -                            
                           fe_values.shape_value (i, q_point) *
                           volumetric_term.value (fe_values.quadrature_point (q_point)) 
                          )
                          *fe_values.JxW (q_point);
      
    cell->get_dof_indices (local_dof_indices);
    constraints.distribute_local_to_global (local_f, local_dof_indices, nonlinear_residual);
    }
*/
 
}

//------------------------------------------------------

template <int dim>
void SS_HC<dim>::assemble_matrix (unsigned int option) 
{  
  // reinit the matrix
  system_matrix.reinit(sparsity_pattern);
  
  QGauss<dim>  quadrature_formula(2);

  const VolumetricTerm<dim> volumetric_term;

  FEValues<dim> fe_values (fe, quadrature_formula, 
                           update_values   | update_gradients |
                           update_quadrature_points | update_JxW_values);

  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();

  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhs (dofs_per_cell);


  std::vector<double>          local_solution_values(n_q_points);
  std::vector<Tensor<1, dim> > local_solution_gradients(n_q_points);

  std::vector<double> conductivity_values(n_q_points);
  std::vector<double> conductivity_derivatives(n_q_points);

  std::vector<unsigned int> local_dof_indices (dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
  
  for (; cell!=endc; ++cell)
    {
    fe_values.reinit (cell);
    cell_matrix = 0;
    fe_values.get_function_values   (solution,local_solution_values   );
    fe_values.get_function_gradients(solution,local_solution_gradients);
    compute_thermal_conductivity(local_solution_values,conductivity_values,conductivity_derivatives);

    for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
      for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
        for (unsigned int j=0; j<dofs_per_cell; ++j)
          {
          cell_matrix(i,j) += (fe_values.shape_grad (i, q_point) *
                               conductivity_values[q_point]      *
                               fe_values.shape_grad (j, q_point) *
                               fe_values.JxW (q_point)
                              );

           if(option==1) // exact jacobian
             cell_matrix(i,j) += (fe_values.shape_grad (i, q_point) *
                                  conductivity_derivatives[q_point] *
                                  fe_values.shape_value(j, q_point) *
                                  local_solution_gradients[q_point] *
                                  fe_values.JxW (q_point)
                                 );
        }
      }
      
      cell->get_dof_indices (local_dof_indices);
      constraints.distribute_local_to_global (cell_matrix, local_dof_indices, system_matrix );     
  }

}

//------------------------------------------------------

template <int dim>
std::pair<unsigned int, double> SS_HC<dim>::linear_solve (Vector<double> &newton_update)
{
  SolverControl solver_control (1000, 1.0e-10);
  SolverGMRES<Vector<double> > gmres (solver_control);
  PreconditionIdentity prec;
  prec.initialize(system_matrix);

  bool matrix_free = true;
  if (!matrix_free) {
    gmres.solve (system_matrix, newton_update, nonlinear_residual, prec);
  } else {
    compute_nonlinear_residual<dim>(dof_handler, constraints, solution, minus_unperturbed_nonlinear_residual);
    minus_unperturbed_nonlinear_residual *= -1.0;
    ActionOfJacobianOnVector<dim> jacobian_apply(dof_handler, constraints, solution, minus_unperturbed_nonlinear_residual, nonlinear_residual);
    gmres.solve (jacobian_apply, newton_update, minus_unperturbed_nonlinear_residual, prec);
 } // end if
  
  std::cout << "   " << solver_control.last_step()
            << " GMRES iterations needed to obtain convergence."
            << std::endl;

  return std::pair<unsigned int, double> ( solver_control.last_step(), solver_control.last_value() ); 
}

//------------------------------------------------------

template <int dim>
void SS_HC<dim>::output_results () const
{
  DataOut<dim> data_out;

  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (solution, "solution");

  data_out.build_patches ();

  std::string str = "solution-"+Utilities::int_to_string(dim)+"d.vtk";
  std::ofstream output (str.c_str());
  data_out.write_vtk (output);
}

//------------------------------------------------------

template <int dim>
void SS_HC<dim>::run () 
{
  std::cout << "Solving problem in " << dim << " space dimensions." << std::endl;
  
  make_grid();
  setup_system ();

  // pick initial guess
  const double initial_flat_value = 33.0;
  VectorTools::interpolate(dof_handler,
                           ConstantFunction<dim>(initial_flat_value), 
                           solution);
  // impose BC on the solution vector
  std::map<unsigned int,double> boundary_values;
  VectorTools::interpolate_boundary_values (dof_handler,
                                            0,  //jcr check this
                                            BoundaryValues<dim>(),
                                            boundary_values);
  for (std::map<unsigned int, double>::const_iterator it = boundary_values.begin(); it != boundary_values.end(); ++it)
    solution(it->first) = (it->second) ;

  // newton data
  const double rtol = 1.0E-11;
  int nonlin_iter = 0;
  bool  newton_convergence = false;
  const double damping = 1.0;
  double residual_norm;
  // compute initial residual
  compute_residual();
  const double initial_residual_norm = nonlinear_residual.l2_norm();

  compute_nonlinear_residual<dim>(dof_handler, constraints, solution, minus_unperturbed_nonlinear_residual);
  // difference of residuals, check
  nonlinear_residual -= minus_unperturbed_nonlinear_residual;
  std::cout << "difference of residual " << nonlinear_residual.l2_norm() << std::endl;

  minus_unperturbed_nonlinear_residual *= -1.0;
  ActionOfJacobianOnVector<dim> jacobian_apply(dof_handler, constraints, solution, minus_unperturbed_nonlinear_residual, nonlinear_residual);
  Vector<double> v, v1, v2;
  v.reinit(dof_handler.n_dofs());
  v1.reinit(dof_handler.n_dofs());
  v2.reinit(dof_handler.n_dofs());
  v=0.0;
  v(3)=1.0;
  jacobian_apply.vmult(v1,v);
  std::cout << "(F(X+eps.v)-F(x))/eps =" ;
  v1.print(std::cout, 15);
  std::cout << std::endl;

  assemble_matrix (0);
  system_matrix.print_formatted(std::cout, 5, true, 0, "0", 1);
  system_matrix.vmult(v2,v);
  std::cout << "Av=" ;
  v2.print(std::cout, 15);
  std::cout << std::endl;

  v.equ(1.0,v1,-1.0,v2);
  std::cout << "norm of the vmult results v1-v2  v1  v2  " << v.l2_norm() << "  "<< v1.l2_norm()<<"  "<<v2.l2_norm()<< std::endl;

  Assert(false, ExcNotImplemented());

  while ( nonlin_iter < 100 && ! newton_convergence ) {
    std::cout <<  "Newton iteration # " << nonlin_iter << "\t:";

    // assemble jacobian
    assemble_matrix (0);
    // zero out the update vector
    newton_update = 0.0;
    // save the unperturbed nonlinear reisudal to pass to MF GMRes
    minus_unperturbed_nonlinear_residual = nonlinear_residual;
    // make this the rhs of the linear system J delta = -f
    nonlinear_residual *= -1.0; // if matrix-free, we do not want this anymore
    // solve the linear system J delta = -f 
    std::pair<unsigned int, double> convergence = linear_solve (newton_update);
    // update Newton solution
    solution.add(damping, newton_update);
  
    // compute residual
    compute_residual();
    residual_norm = nonlinear_residual.l2_norm();
    //    
    std::printf("   %-16.3e %04d        %-5.2e %-5.2e\n", residual_norm, convergence.first, convergence.second, newton_update.l2_norm() );
    newton_convergence = (residual_norm/initial_residual_norm) < rtol ;
    if(newton_convergence)
       std::cout << "  --- Newton has converged --- " << std::endl;
    // increment iteration counter
    ++nonlin_iter;
  } // Newton's loop
      
        
  // output vtk file 
  output_results ();

}

//------------------------------------------------------

int main () 
{
  deallog.depth_console (0);
  const int my_dim = 1;
  SS_HC<my_dim> laplace_problem;
  laplace_problem.run ();
  
  return 0;
}

