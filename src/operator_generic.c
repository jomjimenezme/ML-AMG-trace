/*
 * Copyright (C) 2016, Matthias Rottmann, Artur Strebel, Simon Heybrock, Simone Bacchio, Bjoern Leder.
 * 
 * This file is part of the DDalphaAMG solver library.
 * 
 * The DDalphaAMG solver library is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * The DDalphaAMG solver library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * 
 * You should have received a copy of the GNU General Public License
 * along with the DDalphaAMG solver library. If not, see http://www.gnu.org/licenses/.
 * 
 */

#include "main.h"

void operator_double_init( operator_double_struct *op ) {
  
  op->prnT = NULL;
  op->index_table = NULL;
  op->neighbor_table = NULL;
  op->backward_neighbor_table = NULL;
  op->translation_table = NULL;
  op->D = NULL;
  op->D_vectorized = NULL;
  op->D_transformed_vectorized = NULL;
  op->clover = NULL;
  op->clover_oo_inv = NULL;
  op->clover_vectorized = NULL;
  op->clover_oo_inv_vectorized = NULL;
  op->m0 = 0;
#ifdef HAVE_TM
  op->mu = 0;
  op->mu_even_shift = 0;
  op->mu_odd_shift = 0;
  op->odd_proj = NULL;
  op->tm_term = NULL;
#endif
#ifdef HAVE_TM1p1
  op->epsbar = 0;
  op->epsbar_ig5_even_shift = 0;
  op->epsbar_ig5_odd_shift = 0;
  op->epsbar_term = NULL;
  op->clover_doublet_oo_inv = NULL;
  op->clover_doublet_vectorized = NULL;
  op->clover_doublet_oo_inv_vectorized = NULL;
#endif
  
  for ( int mu=0; mu<4; mu++ )
    op->config_boundary_table[mu] = NULL;
  
  for ( int i=0; i<8; i++ ) {
    op->c.boundary_table[i] = NULL;
    op->c.buffer[i] = NULL;
    op->c.in_use[i] = 0;
  }
  op->c.comm = 1;
  op->buffer = NULL;
}


void operator_double_alloc_projection_buffers( operator_double_struct *op, level_struct *l ) {

  // when used as preconditioner we usually do not need the projection buffers, unless
  // g.method >= 4: then oddeven_setup_double() is called in init.c, method_setup().
  if ( l->depth == 0 ) {
    int its = (l->num_lattice_site_var/2)*l->num_lattice_sites;
#ifdef HAVE_TM1p1
    its *= 2;
#endif
    MALLOC( op->prnT, complex_double, its*8 );
    op->prnZ = op->prnT + its; op->prnY = op->prnZ + its; op->prnX = op->prnY + its;
    op->prpT = op->prnX + its; op->prpZ = op->prpT + its; op->prpY = op->prpZ + its; op->prpX = op->prpY + its;
  }
}
void operator_double_free_projection_buffers( operator_double_struct *op, level_struct *l ) {

  if ( l->depth == 0 ) {
    int its = (l->num_lattice_site_var/2)*l->num_lattice_sites;
#ifdef HAVE_TM1p1
    its *= 2;
#endif
    FREE( op->prnT, complex_double, its*8 );
  }
}

void operator_double_alloc( operator_double_struct *op, const int type, level_struct *l ) {
  
/*********************************************************************************
* Allocates space for setting up an operator.
* - operator_double_struct *op: operator struct for which space is allocated.
* - const int type: Defines the data layout type of the operator.
* Possible values are: { _ORDINARY, _SCHWARZ }
*********************************************************************************/

  int mu, nu, its = 1, its_boundary, nls, clover_site_size, coupling_site_size;
  
  if ( l->depth == 0 ) {
    clover_site_size = 42;
    coupling_site_size = 4*9;
  } else {
    clover_site_size = (l->num_lattice_site_var*(l->num_lattice_site_var+1))/2;
    coupling_site_size = 4*l->num_lattice_site_var*l->num_lattice_site_var;
  }
  
  if ( type ==_SCHWARZ ) {
    its_boundary = 2;
  } else {
    its_boundary = 1;
  }
  for ( mu=0; mu<4; mu++ ) {
    its *= (l->local_lattice[mu]+its_boundary);
  }
  
  nls = (type==_SCHWARZ) ? (2*l->num_lattice_sites-l->num_inner_lattice_sites):l->num_inner_lattice_sites;

  MALLOC( op->D, complex_double, coupling_site_size*nls );
  MALLOC( op->clover, complex_double, clover_site_size*l->num_inner_lattice_sites );

  int block_site_size = ( l->depth == 0 ) ? 12 : (l->num_lattice_site_var/2*(l->num_lattice_site_var/2+1));
  MALLOC( op->odd_proj, complex_double, block_site_size*l->num_inner_lattice_sites );
#ifdef HAVE_TM
  MALLOC( op->tm_term, complex_double, block_site_size*l->num_inner_lattice_sites );
#endif
#ifdef HAVE_TM1p1
  MALLOC( op->epsbar_term, complex_double, block_site_size*l->num_inner_lattice_sites );
#endif

  MALLOC( op->index_table, int, its );
  if ( type ==_ODDEVEN ) {
    MALLOC( op->neighbor_table, int, 5*its );
    MALLOC( op->backward_neighbor_table, int, 5*its );
  } else {
    MALLOC( op->neighbor_table, int, (l->depth==0?4:5)*l->num_inner_lattice_sites );
    MALLOC( op->backward_neighbor_table, int, (l->depth==0?4:5)*l->num_inner_lattice_sites );
  }
  MALLOC( op->translation_table, int, l->num_inner_lattice_sites );

  if ( type == _SCHWARZ && l->depth == 0 && g.odd_even ) {
#ifndef OPTIMIZED_SELF_COUPLING_double

    if( g.csw ) {
#ifdef HAVE_TM //we use LU here
      MALLOC( op->clover_oo_inv, complex_double, 72*(l->num_inner_lattice_sites/2+1) );
#else
      MALLOC( op->clover_oo_inv, complex_double, clover_site_size*(l->num_inner_lattice_sites/2+1) );
#endif
    }
#ifdef HAVE_TM1p1
    MALLOC( op->clover_doublet_oo_inv, complex_double, 12*12*2*(l->num_inner_lattice_sites/2+1) );
#endif

#else
    if( g.csw )
      MALLOC_HUGEPAGES( op->clover_oo_inv_vectorized, double, 144*(l->num_inner_lattice_sites/2+1), 4*SIMD_LENGTH_double );
#ifdef HAVE_TM1p1
    MALLOC_HUGEPAGES( op->clover_doublet_oo_inv_vectorized, double, 2*2*144*(l->num_inner_lattice_sites/2+1), 4*SIMD_LENGTH_double );
#endif

#endif
  }  

  if ( type != _ODDEVEN )
    operator_double_alloc_projection_buffers( op, l );
  
  ghost_alloc_double( 0, &(op->c), l );
  
  for ( mu=0; mu<4; mu++ ) {
    its = 1;
    for ( nu=0; nu<4; nu++ ) {
      if ( mu != nu ) {
        its *= l->local_lattice[nu];
      }
    }
    op->c.num_boundary_sites[2*mu] = its;
    op->c.num_boundary_sites[2*mu+1] = its;
    MALLOC( op->c.boundary_table[2*mu], int, its );
    if ( type == _SCHWARZ ) {
      MALLOC( op->c.boundary_table[2*mu+1], int, its );
      MALLOC( op->config_boundary_table[mu], int, its );
    } else {
      op->c.boundary_table[2*mu+1] = op->c.boundary_table[2*mu];
    }
  }
}


void operator_double_free( operator_double_struct *op, const int type, level_struct *l ) {
  
  int mu, nu, its = 1, clover_site_size, coupling_site_size;
  
  if ( l->depth == 0 ) {
    clover_site_size = 42;
    coupling_site_size = 4*9;
  } else {
    clover_site_size = (l->num_lattice_site_var*(l->num_lattice_site_var+1))/2;
    coupling_site_size = 4*l->num_lattice_site_var*l->num_lattice_site_var;
  }
  
  int its_boundary;
  if ( type ==_SCHWARZ ) {
    its_boundary = 2;
  } else {
    its_boundary = 1;
  }
  for ( mu=0; mu<4; mu++ ) {
    its *= (l->local_lattice[mu]+its_boundary);
  }
  
  int nls = (type==_SCHWARZ) ? (2*l->num_lattice_sites-l->num_inner_lattice_sites) : l->num_inner_lattice_sites;
  FREE( op->D, complex_double, coupling_site_size*nls );
  FREE( op->clover, complex_double, clover_site_size*l->num_inner_lattice_sites );

  int block_site_size = ( l->depth == 0 ) ? 12 : (l->num_lattice_site_var/2*(l->num_lattice_site_var/2+1));
  FREE( op->odd_proj, complex_double, block_site_size*l->num_inner_lattice_sites );
#ifdef HAVE_TM
  FREE( op->tm_term, complex_double, block_site_size*l->num_inner_lattice_sites );
#endif
  if ( type == _SCHWARZ && l->depth == 0 && g.odd_even ) {
#ifndef OPTIMIZED_SELF_COUPLING_double

    if( g.csw ) {
#ifdef HAVE_TM //we use LU here
      FREE( op->clover_oo_inv, complex_double, 72*(l->num_inner_lattice_sites/2+1) );
#else
      FREE( op->clover_oo_inv, complex_double, clover_site_size*(l->num_inner_lattice_sites/2+1) );
#endif
    }
#ifdef HAVE_TM1p1
    FREE( op->clover_doublet_oo_inv, complex_double, 12*12*2*(l->num_inner_lattice_sites/2+1) );
#endif

#else
    if( g.csw )
      FREE_HUGEPAGES( op->clover_oo_inv_vectorized, double, 144*(l->num_inner_lattice_sites/2+1) );
#ifdef HAVE_TM1p1
    FREE_HUGEPAGES( op->clover_doublet_oo_inv_vectorized, double, 2*2*144*(l->num_inner_lattice_sites/2+1) );
#endif

#endif
  }  

#ifdef HAVE_TM1p1
  FREE( op->epsbar_term, complex_double, block_site_size*l->num_inner_lattice_sites );
#endif
  FREE( op->index_table, int, its );
  if ( type ==_ODDEVEN ) {
    FREE( op->neighbor_table, int, 5*its );
    FREE( op->backward_neighbor_table, int, 5*its );
  } else {
    FREE( op->neighbor_table, int, (l->depth==0?4:5)*l->num_inner_lattice_sites );
    FREE( op->backward_neighbor_table, int, (l->depth==0?4:5)*l->num_inner_lattice_sites );
  }
  FREE( op->translation_table, int, l->num_inner_lattice_sites );
  
  if ( type != _ODDEVEN )
    operator_double_free_projection_buffers( op, l );
  
  ghost_free_double( &(op->c), l );
  
  for ( mu=0; mu<4; mu++ ) {
    its = 1;
    for ( nu=0; nu<4; nu++ ) {
      if ( mu != nu ) {
        its *= l->local_lattice[nu];
      }
    }
    
    FREE( op->c.boundary_table[2*mu], int, its );
    if ( type == _SCHWARZ ) {
      FREE( op->c.boundary_table[2*mu+1], int, its );
      FREE( op->config_boundary_table[mu], int, its );
    } else {
      op->c.boundary_table[2*mu+1] = NULL;
    }
  }
}


void operator_double_define( operator_double_struct *op, level_struct *l ) {
  
  int i, mu, t, z, y, x, *it = op->index_table,
    ls[4], le[4], l_st[4], l_en[4], *dt = op->table_dim;
  
  for ( mu=0; mu<4; mu++ ) {
    dt[mu] = l->local_lattice[mu]+1;
    ls[mu] = 0;
    le[mu] = ls[mu] + l->local_lattice[mu];
    l_st[mu] = ls[mu];
    l_en[mu] = le[mu];
  }
  
  // define index table
  // lexicographic inner cuboid and
  // lexicographic +T,+Z,+Y,+X boundaries
  i=0;
  // inner hyper cuboid
  for ( t=ls[T]; t<le[T]; t++ )
    for ( z=ls[Z]; z<le[Z]; z++ )
      for ( y=ls[Y]; y<le[Y]; y++ )
        for ( x=ls[X]; x<le[X]; x++ ) {
          it[ lex_index( t, z, y, x, dt ) ] = i; i++;
        }
  // boundaries (buffers)
  for ( mu=0; mu<4; mu++ ) {
    l_st[mu] = le[mu];
    l_en[mu] = le[mu]+1;
    
    for ( t=l_st[T]; t<l_en[T]; t++ )
      for ( z=l_st[Z]; z<l_en[Z]; z++ )
        for ( y=l_st[Y]; y<l_en[Y]; y++ )
          for ( x=l_st[X]; x<l_en[X]; x++ ) {
            it[ lex_index( t, z, y, x, dt ) ] = i; i++;
          }
          
    l_st[mu] = ls[mu];
    l_en[mu] = le[mu];
  }
  
  // define neighbor table (for the application of the entire operator),
  // negative inner boundary table (for communication),
  // translation table (for translation to lexicographical site ordnering)
  define_nt_bt_tt( op->neighbor_table, op->backward_neighbor_table, op->c.boundary_table, op->translation_table, it, dt, l );
}

void operator_double_set_couplings( operator_double_struct *op, level_struct *l ) {

  operator_double_set_self_couplings( op, l );
  operator_double_set_neighbor_couplings( op, l );

}

void operator_double_set_neighbor_couplings( operator_double_struct *op, level_struct *l ) {

#ifdef OPTIMIZED_NEIGHBOR_COUPLING_double
  int i, n = 2*l->num_lattice_sites - l->num_inner_lattice_sites;

  for ( i=0; i<n; i++ ) {
    double *D_vectorized = op->D_vectorized + 96*i;
    double *D_transformed_vectorized = op->D_transformed_vectorized + 96*i;
    complex_double *D_pt = op->D + 36*i;
    for ( int mu=0; mu<4; mu++ )
      set_double_D_vectorized( D_vectorized+24*mu, D_transformed_vectorized+24*mu, D_pt+9*mu );
  }
#endif

}

void operator_double_set_self_couplings( operator_double_struct *op, level_struct *l ) {

#ifdef OPTIMIZED_SELF_COUPLING_double
  int i, n = l->num_inner_lattice_sites;
  
  if ( g.csw != 0 )
    for ( i=0; i<n; i++ ) {
      double *clover_vectorized_pt = op->clover_vectorized + 144*i;
      config_double clover_pt = op->clover + 42*i;
      sse_set_clover_double( clover_vectorized_pt, clover_pt );
#ifdef HAVE_TM1p1
      double *clover_doublet_vectorized_pt = op->clover_doublet_vectorized + 288*i;
      sse_set_clover_doublet_double( clover_doublet_vectorized_pt, clover_pt );
#endif
#ifdef HAVE_TM
      config_double tm_term_pt = op->tm_term + 12*i;
      sse_add_diagonal_clover_double( clover_vectorized_pt, tm_term_pt );
#ifdef HAVE_TM1p1
      sse_add_diagonal_clover_doublet_double( clover_doublet_vectorized_pt, tm_term_pt );
#endif
#endif
    }
#endif
  
}

void operator_double_test_routine( operator_double_struct *op, level_struct *l, struct Thread *threading ) {

/*********************************************************************************
* Checks for correctness of operator data layout by doing:
* - Applying D_W in double precision to a double vector.
* - Translates the same vector into double and apply D_W in double to this
*   vector and translate it back to double
* - Compare solutions ( Difference should be close to 0 ).
* If enabled, also tests odd even preconditioning.
*********************************************************************************/ 

  int ivs = l->inner_vector_size;
  double diff;
  
  vector_double vd1=NULL, vd2, vd3, vd4;
  vector_double vp1=NULL, vp2;

  PUBLIC_MALLOC( vd1, complex_double, 4*ivs );
  PUBLIC_MALLOC( vp1, complex_double, 2*ivs );

  vd2 = vd1 + ivs; vd3 = vd2 + ivs; vd4 = vd3 + ivs; vp2 = vp1 + ivs;

  START_LOCKED_MASTER(threading)
  
  vector_double_define_random( vd1, 0, l->inner_vector_size, l );
  apply_operator_double( vd2, vd1, &(g.p), l, no_threading );
  
  trans_double( vp1, vd1, op->translation_table, l, no_threading );
  apply_operator_double( vp2, vp1, &(l->p_double), l, no_threading );
  trans_back_double( vd3, vp2, op->translation_table, l, no_threading );
  
  vector_double_minus( vd4, vd3, vd2, 0, l->inner_vector_size, l );
  diff = global_norm_double( vd4, 0, ivs, l, no_threading )/
    global_norm_double( vd3, 0, ivs, l, no_threading );

  test0_double("depth: %d, correctness of schwarz double Dirac operator: %le\n", l->depth, diff );
  END_LOCKED_MASTER(threading)

  if(threading->n_core > 1) {
    apply_operator_double( vp2, vp1, &(l->p_double), l, threading );

    SYNC_MASTER_TO_ALL(threading)
    SYNC_CORES(threading)

    START_LOCKED_MASTER(threading)
    trans_back_double( vd3, vp2, op->translation_table, l, no_threading );
    vector_double_minus( vd4, vd3, vd2, 0, l->inner_vector_size, l );
    diff = global_norm_double( vd4, 0, ivs, l, no_threading ) /
      global_norm_double( vd3, 0, ivs, l, no_threading );

    if ( diff > EPS_double )
      printf0("\x1b[31m");
    printf0("depth: %d, correctness of schwarz double Dirac operator with threading: %le\n", l->depth, diff );
    if ( diff > EPS_double )
      printf0("\x1b[0m");
    if(diff > g.test) g.test = diff;

    END_LOCKED_MASTER(threading) 
  }    
  
  PUBLIC_FREE( vd1, complex_double, 4*ivs );
  PUBLIC_FREE( vp1, complex_double, 2*ivs );

  START_LOCKED_MASTER(threading)
  if ( g.method >=4 && g.odd_even )
    oddeven_double_test( l );
  END_LOCKED_MASTER(threading) 
}
