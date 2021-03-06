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

#ifndef DIRAC_SSE_H
#define DIRAC_SSE_H
#ifdef SSE

void prp_double( complex_double *prn[4], complex_double *phi, int start, int end );
void prp_double( complex_double *prn[4], complex_double *phi, int start, int end );
void prn_su3_double( complex_double *prp[4], complex_double *phi, operator_double_struct *op, int *neighbor, int start, int end );
void prn_su3_double( complex_double *prp[4], complex_double *phi, operator_double_struct *op, int *neighbor, int start, int end );
void pbn_double( complex_double *eta, complex_double *prp[4], int start, int end );
void pbn_double( complex_double *eta, complex_double *prp[4], int start, int end );
void su3_pbp_double( complex_double* eta, complex_double *prn[4], operator_double_struct *op, int *neighbor, int start, int end );
void su3_pbp_double( complex_double* eta, complex_double *prn[4], operator_double_struct *op, int *neighbor, int start, int end );

void dprp_double( complex_double *prn[4], complex_double *phi, int start, int end );
void dprp_double( complex_double *prn[4], complex_double *phi, int start, int end );
void dprn_su3_double( complex_double *prp[4], complex_double *phi, operator_double_struct *op, int *neighbor, int start, int end );
void dprn_su3_double( complex_double *prp[4], complex_double *phi, operator_double_struct *op, int *neighbor, int start, int end );
void dpbn_double( complex_double *eta, complex_double *prp[4], int start, int end );
void dpbn_double( complex_double *eta, complex_double *prp[4], int start, int end );
void su3_dpbp_double( complex_double* eta, complex_double *prn[4], operator_double_struct *op, int *neighbor, int start, int end );
void su3_dpbp_double( complex_double* eta, complex_double *prn[4], operator_double_struct *op, int *neighbor, int start, int end );

void block_oddeven_plus_coupling_double( double *eta, double *D, double *phi, int mu,
                                         int start, int end, int *ind, int *neighbor );
void block_oddeven_plus_coupling_double( double *eta, double *D, double *phi, int mu,
                                        int start, int end, int *ind, int *neighbor );
void block_oddeven_minus_coupling_double( double *eta, double *D, double *phi, int mu,
                                          int start, int end, int *ind, int *neighbor );
void block_oddeven_minus_coupling_double( double *eta, double *D, double *phi, int mu,
                                         int start, int end, int *ind, int *neighbor );
void block_oddeven_nplus_coupling_double( double *eta, double *D, double *phi, int mu,
                                         int start, int end, int *ind, int *neighbor );
void block_oddeven_nplus_coupling_double( double *eta, double *D, double *phi, int mu,
                                        int start, int end, int *ind, int *neighbor );
void block_oddeven_nminus_coupling_double( double *eta, double *D, double *phi, int mu,
                                          int start, int end, int *ind, int *neighbor );
void block_oddeven_nminus_coupling_double( double *eta, double *D, double *phi, int mu,
                                         int start, int end, int *ind, int *neighbor );
void boundary_minus_coupling_double( double *eta, double *D, double *phi, int mu,
                                              int start, int end, int *ind, int *neighbor );
void boundary_minus_coupling_double( double *eta, double *D, double *phi, int mu,
                                             int start, int end, int *ind, int *neighbor );
void boundary_plus_coupling_double( double *eta, double *D, double *phi, int mu,
                                             int start, int end, int *ind, int *neighbor );
void boundary_plus_coupling_double( double *eta, double *D, double *phi, int mu,
                                            int start, int end, int *ind, int *neighbor );
void boundary_nminus_coupling_double( double *eta, double *D, double *phi, int mu,
                                              int start, int end, int *ind, int *neighbor );
void boundary_nminus_coupling_double( double *eta, double *D, double *phi, int mu,
                                             int start, int end, int *ind, int *neighbor );
void boundary_nplus_coupling_double( double *eta, double *D, double *phi, int mu,
                                             int start, int end, int *ind, int *neighbor );
void boundary_nplus_coupling_double( double *eta, double *D, double *phi, int mu,
                                            int start, int end, int *ind, int *neighbor );

void sse_set_clover_double( double *out, complex_double *in );
void sse_set_clover_double( double *out, complex_double *in );
void sse_set_clover_doublet_double( double *out, complex_double *in );
void sse_set_clover_doublet_double( double *out, complex_double *in );
void sse_add_diagonal_clover_double( double *out, complex_double *diag );
void sse_add_diagonal_clover_double( double *out, complex_double *diag );
void sse_add_diagonal_clover_doublet_double( double *out, complex_double *diag );
void sse_add_diagonal_clover_doublet_double( double *out, complex_double *diag );
void sse_clover_double( vector_double eta, vector_double phi, operator_double_struct *op, int start, int end, level_struct *l, struct Thread *threading );
void sse_clover_double( vector_double eta, vector_double phi, operator_double_struct *op, int start, int end, level_struct *l, struct Thread *threading );
void sse_site_clover_double( double *eta, const double *phi, const double *clover );
void sse_site_clover_double( double *eta, const double *phi, double *clover );
void sse_site_clover_doublet_double( double *eta, const double *phi, const double *clover );
void sse_site_clover_doublet_double( double *eta, const double *phi, double *clover );

void sse_site_clover_invert_double( double *clover_in, double *clover_out );
void sse_site_clover_invert_double( double *clover_in, double *clover_out );
void sse_site_clover_doublet_invert_double( double *clover_in, config_double eps_term, double *clover_out );
void sse_site_clover_doublet_invert_double( double *clover_in, config_double eps_term, double *clover_out );


static inline void sse_mvm_double_simd_length( const complex_double *eta, const complex_double *D, const complex_double *phi ) {}

static inline void sse_mvm_double_simd_length(
    const complex_double *eta, const complex_double *D, const complex_double *phi ) {
#ifdef SSE
  __m128 gauge_re;
  __m128 gauge_im;
  __m128 in_re[3];
  __m128 in_im[3];
  __m128 out_re[3];
  __m128 out_im[3];

  int elements = SIMD_LENGTH_double;

  // j runs over all right-hand sides, using vectorization
  for(int i=0; i<3; i++) {
    in_re[i] = _mm_load_ps((double *)(phi+i*elements));
    in_im[i] = _mm_load_ps((double *)(phi+i*elements)+elements);
  }

  for(int i=0; i<3; i++) {
    gauge_re = _mm_set1_ps(creal(D[0+3*i]));
    gauge_im = _mm_set1_ps(cimag(D[0+3*i]));
    cmul(gauge_re, gauge_im, in_re[0], in_im[0], out_re+i, out_im+i);
    gauge_re = _mm_set1_ps(creal(D[1+3*i]));
    gauge_im = _mm_set1_ps(cimag(D[1+3*i]));
    cfmadd(gauge_re, gauge_im, in_re[1], in_im[1], out_re+i, out_im+i);
    gauge_re = _mm_set1_ps(creal(D[2+3*i]));
    gauge_im = _mm_set1_ps(cimag(D[2+3*i]));
    cfmadd(gauge_re, gauge_im, in_re[2], in_im[2], out_re+i, out_im+i);
  }

  for(int i=0; i<3; i++) {
    _mm_store_ps((double *)(eta+i*elements),          out_re[i]);
    _mm_store_ps((double *)(eta+i*elements)+elements, out_im[i]);
  }
#endif
}


static inline void sse_mvm_double( const complex_double *eta, const complex_double *D,
                                   const complex_double *phi, int elements ) {}
// spinors are vectorized, gauge is same for all (use for multiple rhs)
static inline void sse_mvm_double( const complex_double *eta, const complex_double *D,
                                  const complex_double *phi, int elements ) {
#ifdef SSE
  __m128 gauge_re;
  __m128 gauge_im;
  __m128 in_re[3];
  __m128 in_im[3];
  __m128 out_re[3];
  __m128 out_im[3];

  // j runs over all right-hand sides, using vectorization
  for(int j=0; j<elements; j+=SIMD_LENGTH_double) {
    for(int i=0; i<3; i++) {
      in_re[i] = _mm_load_ps((double *)(phi+i*elements)+j);
      in_im[i] = _mm_load_ps((double *)(phi+i*elements)+j+elements);
    }

    for(int i=0; i<3; i++) {
      gauge_re = _mm_set1_ps(creal(D[0+3*i]));
      gauge_im = _mm_set1_ps(cimag(D[0+3*i]));
      cmul(gauge_re, gauge_im, in_re[0], in_im[0], out_re+i, out_im+i);
      gauge_re = _mm_set1_ps(creal(D[1+3*i]));
      gauge_im = _mm_set1_ps(cimag(D[1+3*i]));
      cfmadd(gauge_re, gauge_im, in_re[1], in_im[1], out_re+i, out_im+i);
      gauge_re = _mm_set1_ps(creal(D[2+3*i]));
      gauge_im = _mm_set1_ps(cimag(D[2+3*i]));
      cfmadd(gauge_re, gauge_im, in_re[2], in_im[2], out_re+i, out_im+i);
    }

    for(int i=0; i<3; i++) {
      _mm_store_ps((double *)(eta+i*elements)+j,          out_re[i]);
      _mm_store_ps((double *)(eta+i*elements)+j+elements, out_im[i]);
    }
  }
#endif
}


static inline void sse_mvmh_double( const complex_double *eta, const complex_double *D,
                                    const complex_double *phi, int elements ) {}
// spinors are vectorized, gauge is same for all (use for multiple rhs)
static inline void sse_mvmh_double( const complex_double *eta, const complex_double *D,
                                   const complex_double *phi, int elements ) {
#ifdef SSE
  __m128 gauge_re;
  __m128 gauge_im;
  __m128 in_re[3];
  __m128 in_im[3];
  __m128 out_re[3];
  __m128 out_im[3];


  // j runs over all right-hand sides, using vectorization
  for(int j=0; j<elements; j+=SIMD_LENGTH_double) {
    for(int i=0; i<3; i++) {
      in_re[i] = _mm_load_ps((double *)(phi+i*elements)+j);
      in_im[i] = _mm_load_ps((double *)(phi+i*elements)+j+elements);
    }

    for(int i=0; i<3; i++) {
      gauge_re = _mm_set1_ps(creal(D[0+i]));
      gauge_im = _mm_set1_ps(cimag(D[0+i]));
      cmul_conj(gauge_re, gauge_im, in_re[0], in_im[0], out_re+i, out_im+i);
      gauge_re = _mm_set1_ps(creal(D[3+i]));
      gauge_im = _mm_set1_ps(cimag(D[3+i]));
      cfmadd_conj(gauge_re, gauge_im, in_re[1], in_im[1], out_re+i, out_im+i);
      gauge_re = _mm_set1_ps(creal(D[6+i]));
      gauge_im = _mm_set1_ps(cimag(D[6+i]));
      cfmadd_conj(gauge_re, gauge_im, in_re[2], in_im[2], out_re+i, out_im+i);
    }

    for(int i=0; i<3; i++) {
      _mm_store_ps((double *)(eta+i*elements)+j,          out_re[i]);
      _mm_store_ps((double *)(eta+i*elements)+j+elements, out_im[i]);
    }
  }
#endif
}


static inline void sse_twospin_double( complex_double *out_spin0and1, complex_double *out_spin2and3, const complex_double *in, int elements, int mu, double sign ) {}
// mu is according to the enum for T,Z,Y,X defined in clifford.h
static inline void sse_twospin_double( complex_double *out_spin0and1, complex_double *out_spin2and3, const complex_double *in, int elements, int mu, double sign ) {

#ifdef SSE
  __m128 scale_re;
  __m128 scale_im;
  complex_double *out;
  // components 0-5  are subtracted from out_spin0_and1
  // components 6-11 are subtracted from out_spin2_and3

  // 6 complex components = 12
  for(int i=0; i<12*elements; i+=SIMD_LENGTH_double) {
    __m128 tmp = _mm_load_ps((double *)in            + i);
    __m128 out = _mm_load_ps((double *)out_spin0and1 + i);

    out = _mm_sub_ps(out, tmp);

    _mm_store_ps((double *)out_spin0and1 + i, out);
  }
  for(int i=12*elements; i<24*elements; i+=SIMD_LENGTH_double) {
    __m128 tmp = _mm_load_ps((double *)in            + i);
    __m128 out = _mm_load_ps((double *)out_spin2and3 + i);

    out = _mm_sub_ps(out, tmp);

    _mm_store_ps((double *)out_spin2and3 + i, out);
  }

  out = out_spin2and3;
  for(int spin=0; spin<4; spin++) {
    if(spin == 2)
      out = out_spin0and1;
    scale_re = _mm_set1_ps(sign*creal(gamma_val[mu][spin]));
    scale_im = _mm_set1_ps(sign*cimag(gamma_val[mu][spin]));
    // factors of 2 are for complex
    for(int j=0; j<3; j++) {
      for(int i=0; i<elements; i+=SIMD_LENGTH_double) {
        __m128 in_re  = _mm_load_ps((double *)in  + i + (2*(3*gamma_co[mu][spin]+j)+0)*elements);
        __m128 in_im  = _mm_load_ps((double *)in  + i + (2*(3*gamma_co[mu][spin]+j)+1)*elements);
        __m128 out_re = _mm_load_ps((double *)out + i + (2*(3*spin+j)+0)*elements);
        __m128 out_im = _mm_load_ps((double *)out + i + (2*(3*spin+j)+1)*elements);

        cfmadd(scale_re, scale_im, in_re, in_im, &out_re, &out_im);

        _mm_store_ps((double *)out + i + (2*(3*spin+j)+0)*elements, out_re);
        _mm_store_ps((double *)out + i + (2*(3*spin+j)+1)*elements, out_im);
      }
    }
  }
#endif
}


static inline void sse_twospin2_p_double_simd_length( complex_double *out_spin0and1, complex_double *out_spin2and3, const complex_double *in, int mu ) {}
// mu is according to the enum for T,Z,Y,X defined in clifford.h
static inline void sse_twospin2_p_double_simd_length( complex_double *out_spin0and1, complex_double *out_spin2and3, const complex_double *in, int mu ) {
#ifdef SSE
  __m128 scale_re;
  __m128 scale_im;
//   __m128 out_re;
//   __m128 out_im;
  complex_double *out;
  int elements = SIMD_LENGTH_double;
  // components 0-5  are subtracted from out_spin0_and1
  // components 6-11 are subtracted from out_spin2_and3

  // 6 complex components = 12
  for(int i=0; i<12*elements; i+=SIMD_LENGTH_double) {
    __m128 tmp = _mm_load_ps((double *)in + i);
    _mm_store_ps((double *)out_spin0and1 + i, tmp);
  }
  for(int i=12*elements; i<24*elements; i+=SIMD_LENGTH_double) {
    __m128 tmp = _mm_load_ps((double *)in + i);
    _mm_store_ps((double *)out_spin2and3 + i, tmp);
  }

  out = out_spin2and3;
  for(int spin=0; spin<4; spin++) {
    if(spin == 2)
      out = out_spin0and1;
    scale_re = _mm_set1_ps(-creal(gamma_val[mu][spin]));
    scale_im = _mm_set1_ps(-cimag(gamma_val[mu][spin]));
    // factors of 2 are for complex
    for(int j=0; j<3; j++) {
      __m128 in_re  = _mm_load_ps((double *)in   + (2*(3*gamma_co[mu][spin]+j)+0)*elements);
      __m128 in_im  = _mm_load_ps((double *)in   + (2*(3*gamma_co[mu][spin]+j)+1)*elements);
      __m128 out_re = _mm_load_ps((double *)out  + (2*(3*spin+j)+0)*elements);
      __m128 out_im = _mm_load_ps((double *)out  + (2*(3*spin+j)+1)*elements);

      cmul(scale_re, scale_im, in_re, in_im, &out_re, &out_im);

      _mm_store_ps((double *)out  + (2*(3*spin+j)+0)*elements, out_re);
      _mm_store_ps((double *)out  + (2*(3*spin+j)+1)*elements, out_im);
    }
  }
#endif
}


static inline void sse_twospin2_p_double( complex_double *out_spin0and1, complex_double *out_spin2and3, const complex_double *in, int elements, int mu ) {}
// mu is according to the enum for T,Z,Y,X defined in clifford.h
static inline void sse_twospin2_p_double( complex_double *out_spin0and1, complex_double *out_spin2and3, const complex_double *in, int elements, int mu ) {

#ifdef SSE
  __m128 scale_re;
  __m128 scale_im;
//   __m128 out_re;
//   __m128 out_im;
  complex_double *out;
  // components 0-5  are subtracted from out_spin0_and1
  // components 6-11 are subtracted from out_spin2_and3

  // 6 complex components = 12
  for(int i=0; i<12*elements; i+=SIMD_LENGTH_double) {
    __m128 tmp = _mm_load_ps((double *)in + i);
    _mm_store_ps((double *)out_spin0and1 + i, tmp);
  }
  for(int i=12*elements; i<24*elements; i+=SIMD_LENGTH_double) {
    __m128 tmp = _mm_load_ps((double *)in + i);
    _mm_store_ps((double *)out_spin2and3 + i, tmp);
  }

  out = out_spin2and3;
  for(int spin=0; spin<4; spin++) {
    if(spin == 2)
      out = out_spin0and1;
    scale_re = _mm_set1_ps(-creal(gamma_val[mu][spin]));
    scale_im = _mm_set1_ps(-cimag(gamma_val[mu][spin]));
    // factors of 2 are for complex
    for(int j=0; j<3; j++) {
      for(int i=0; i<elements; i+=SIMD_LENGTH_double) {
        __m128 in_re  = _mm_load_ps((double *)in  + i + (2*(3*gamma_co[mu][spin]+j)+0)*elements);
        __m128 in_im  = _mm_load_ps((double *)in  + i + (2*(3*gamma_co[mu][spin]+j)+1)*elements);
        __m128 out_re = _mm_load_ps((double *)out + i + (2*(3*spin+j)+0)*elements);
        __m128 out_im = _mm_load_ps((double *)out + i + (2*(3*spin+j)+1)*elements);

        cmul(scale_re, scale_im, in_re, in_im, &out_re, &out_im);

        _mm_store_ps((double *)out + i + (2*(3*spin+j)+0)*elements, out_re);
        _mm_store_ps((double *)out + i + (2*(3*spin+j)+1)*elements, out_im);
      }
    }
  }
#endif
}


static inline void sse_spin0and1_site_clover_double( const complex_double *eta, const complex_double *phi, const config_double clover, double shift, int elements ) {}

static inline void sse_spin0and1_site_clover_double( const complex_double *eta, const complex_double *phi, const config_double clover, double shift, int elements ) {
#ifdef SSE
  // offset computations 2*index+0/1 are for real and imaginary parts

  // diagonal
  if ( g.csw == 0.0 ) {
    for(int i=0; i<elements; i+=SIMD_LENGTH_double) {
      for(int j=0; j<6; j++) {
        __m128 factor = _mm_set1_ps((double)shift);
        __m128 in_re  = _mm_load_ps((double *)phi + i + (2*j+0)*elements);
        __m128 in_im  = _mm_load_ps((double *)phi + i + (2*j+1)*elements);

        in_re = _mm_mul_ps( factor, in_re );
        in_im = _mm_mul_ps( factor, in_im );

        _mm_store_ps((double *)eta + i + (2*j+0)*elements, in_re);
        _mm_store_ps((double *)eta + i + (2*j+1)*elements, in_im);
      }
      __m128 zero = _mm_setzero_ps();
      for(int j=6; j<12; j++) {
        _mm_store_ps((double *)eta + i + (2*j+0)*elements, zero);
        _mm_store_ps((double *)eta + i + (2*j+1)*elements, zero);
      }
    }
  } else {
    for(int i=0; i<elements; i+=SIMD_LENGTH_double) {
      for(int j=0; j<6; j++) {
        __m128 clover_re = _mm_set1_ps(creal(clover[j]));
        __m128 clover_im = _mm_set1_ps(cimag(clover[j]));
        __m128 in_re  = _mm_load_ps((double *)phi + i + (2*j+0)*elements);
        __m128 in_im  = _mm_load_ps((double *)phi + i + (2*j+1)*elements);
        __m128 out_re; __m128 out_im;
        
        cmul(clover_re, clover_im, in_re, in_im, &out_re, &out_im);

        _mm_store_ps((double *)eta + i + (2*j+0)*elements, out_re);
        _mm_store_ps((double *)eta + i + (2*j+1)*elements, out_im);
      }
      __m128 zero = _mm_setzero_ps();
      for(int j=6; j<12; j++) {
        _mm_store_ps((double *)eta + i + (2*j+0)*elements, zero);
        _mm_store_ps((double *)eta + i + (2*j+1)*elements, zero);
      }
    }

    // spin 0 and 1
    __m128 clover_re;
    __m128 clover_im;
    __m128 in_re;
    __m128 in_im;
    __m128 out_re;
    __m128 out_im;
    for(int i=0; i<elements; i+=SIMD_LENGTH_double) {

      // io = index out, ii = index in, ic = index clover
      int ic = 12;
      for(int io=0; io<5; io++) {
        out_re = _mm_load_ps((double *)eta + i + (2*io+0)*elements);
        out_im = _mm_load_ps((double *)eta + i + (2*io+1)*elements);
        for(int ii=io+1; ii<=5; ii++) {
          clover_re = _mm_set1_ps(creal(clover[ic]));
          clover_im = _mm_set1_ps(cimag(clover[ic]));
          ic++;
          in_re  = _mm_load_ps((double *)phi + i + (2*ii+0)*elements);
          in_im  = _mm_load_ps((double *)phi + i + (2*ii+1)*elements);

          cfmadd(clover_re, clover_im, in_re, in_im, &out_re, &out_im);
        }
        _mm_store_ps((double *)eta + i + (2*io+0)*elements, out_re);
        _mm_store_ps((double *)eta + i + (2*io+1)*elements, out_im);
      }

      ic = 12;
      for(int ii=0; ii<5; ii++) {
        in_re  = _mm_load_ps((double *)phi + i + (2*ii+0)*elements);
        in_im  = _mm_load_ps((double *)phi + i + (2*ii+1)*elements);
        for(int io=ii+1; io<=5; io++) {
          clover_re = _mm_set1_ps(creal(clover[ic]));
          clover_im = _mm_set1_ps(cimag(clover[ic]));
          ic++;
          out_re = _mm_load_ps((double *)eta + i + (2*io+0)*elements);
          out_im = _mm_load_ps((double *)eta + i + (2*io+1)*elements);

          cfmadd_conj(clover_re, clover_im, in_re, in_im, &out_re, &out_im);

          _mm_store_ps((double *)eta + i + (2*io+0)*elements, out_re);
          _mm_store_ps((double *)eta + i + (2*io+1)*elements, out_im);
        }
      }
    }
  }
#endif
}

static inline void sse_diagonal_aggregate_double( const complex_double *eta1, const complex_double *eta2, const complex_double *phi, const config_double diag, int elements ) {}

static inline void sse_diagonal_aggregate_double( const complex_double *eta1, const complex_double *eta2, const complex_double *phi, const config_double diag, int elements ) {
#ifdef SSE
  // offset computations 2*index+0/1 are for real and imaginary parts

  // diagonal
  for(int i=0; i<elements; i+=SIMD_LENGTH_double) {
    __m128 zero = _mm_setzero_ps();
    for(int j=0; j<6; j++) {
      __m128 factor = _mm_set1_ps(creal(diag[j]));
      __m128 in_re  = _mm_load_ps((double *)phi + i + (2*j+0)*elements);
      __m128 in_im  = _mm_load_ps((double *)phi + i + (2*j+1)*elements);
      
      in_re = _mm_mul_ps( factor, in_re );
      in_im = _mm_mul_ps( factor, in_im );
      
      _mm_store_ps((double *)eta1 + i + (2*j+0)*elements, in_re);
      _mm_store_ps((double *)eta1 + i + (2*j+1)*elements, in_im);
      _mm_store_ps((double *)eta2 + i + (2*j+0)*elements, zero);
      _mm_store_ps((double *)eta2 + i + (2*j+1)*elements, zero);
    }
    for(int j=6; j<12; j++) {
      __m128 factor = _mm_set1_ps(creal(diag[j]));
      __m128 in_re  = _mm_load_ps((double *)phi + i + (2*j+0)*elements);
      __m128 in_im  = _mm_load_ps((double *)phi + i + (2*j+1)*elements);
      
      in_re = _mm_mul_ps( factor, in_re );
      in_im = _mm_mul_ps( factor, in_im );
      
      _mm_store_ps((double *)eta2 + i + (2*j+0)*elements, in_re);
      _mm_store_ps((double *)eta2 + i + (2*j+1)*elements, in_im);
      _mm_store_ps((double *)eta1 + i + (2*j+0)*elements, zero);
      _mm_store_ps((double *)eta1 + i + (2*j+1)*elements, zero);
    }
  }
#endif
}


static inline void sse_spin2and3_site_clover_double( const complex_double *eta, const complex_double *phi, const config_double clover, double shift, int elements ) {}

static inline void sse_spin2and3_site_clover_double( const complex_double *eta, const complex_double *phi, const config_double clover, double shift, int elements ) {
#ifdef SSE
  // offset computations 2*index+0/1 are for real and imaginary parts

  // diagonal
  if ( g.csw == 0.0 ) {
    for(int i=0; i<elements; i+=SIMD_LENGTH_double) {
      __m128 zero = _mm_setzero_ps();
      for(int j=0; j<6; j++) {
        _mm_store_ps((double *)eta + i + (2*j+0)*elements, zero);
        _mm_store_ps((double *)eta + i + (2*j+1)*elements, zero);
      }
      for(int j=6; j<12; j++) {
        __m128 factor = _mm_set1_ps((double)shift);
        __m128 in_re  = _mm_load_ps((double *)phi + i + (2*j+0)*elements);
        __m128 in_im  = _mm_load_ps((double *)phi + i + (2*j+1)*elements);

        in_re = _mm_mul_ps( factor, in_re );
        in_im = _mm_mul_ps( factor, in_im );

        _mm_store_ps((double *)eta + i + (2*j+0)*elements, in_re);
        _mm_store_ps((double *)eta + i + (2*j+1)*elements, in_im);
      }
    }
    
    
    
  } else {
    for(int i=0; i<elements; i+=SIMD_LENGTH_double) {
      __m128 zero = _mm_setzero_ps();
      for(int j=0; j<6; j++) {
        _mm_store_ps((double *)eta + i + (2*j+0)*elements, zero);
        _mm_store_ps((double *)eta + i + (2*j+1)*elements, zero);
      }
      for(int j=6; j<12; j++) {
        __m128 clover_re = _mm_set1_ps(creal(clover[j]));
        __m128 clover_im = _mm_set1_ps(cimag(clover[j]));
        __m128 in_re  = _mm_load_ps((double *)phi + i + (2*j+0)*elements);
        __m128 in_im  = _mm_load_ps((double *)phi + i + (2*j+1)*elements);
        __m128 out_re; __m128 out_im;

        cmul(clover_re, clover_im, in_re, in_im, &out_re, &out_im);

        _mm_store_ps((double *)eta + i + (2*j+0)*elements, out_re);
        _mm_store_ps((double *)eta + i + (2*j+1)*elements, out_im);
      }
    }

    // spin 0 and 1
    __m128 clover_re;
    __m128 clover_im;
    __m128 in_re;
    __m128 in_im;
    __m128 out_re;
    __m128 out_im;
    for(int i=0; i<elements; i+=SIMD_LENGTH_double) {

      // io = index out, ii = index in, ic = index clover
      int ic = 27;
      for(int io=6; io<11; io++) {
        out_re = _mm_load_ps((double *)eta + i + (2*io+0)*elements);
        out_im = _mm_load_ps((double *)eta + i + (2*io+1)*elements);
        for(int ii=io+1; ii<=11; ii++) {
          clover_re = _mm_set1_ps(creal(clover[ic]));
          clover_im = _mm_set1_ps(cimag(clover[ic]));
          ic++;
          in_re  = _mm_load_ps((double *)phi + i + (2*ii+0)*elements);
          in_im  = _mm_load_ps((double *)phi + i + (2*ii+1)*elements);

          cfmadd(clover_re, clover_im, in_re, in_im, &out_re, &out_im);
        }
        _mm_store_ps((double *)eta + i + (2*io+0)*elements, out_re);
        _mm_store_ps((double *)eta + i + (2*io+1)*elements, out_im);
      }

      ic = 27;
      for(int ii=6; ii<11; ii++) {
        in_re  = _mm_load_ps((double *)phi + i + (2*ii+0)*elements);
        in_im  = _mm_load_ps((double *)phi + i + (2*ii+1)*elements);
        for(int io=ii+1; io<=11; io++) {
          clover_re = _mm_set1_ps(creal(clover[ic]));
          clover_im = _mm_set1_ps(cimag(clover[ic]));
          ic++;
          out_re = _mm_load_ps((double *)eta + i + (2*io+0)*elements);
          out_im = _mm_load_ps((double *)eta + i + (2*io+1)*elements);

          cfmadd_conj(clover_re, clover_im, in_re, in_im, &out_re, &out_im);

          _mm_store_ps((double *)eta + i + (2*io+0)*elements, out_re);
          _mm_store_ps((double *)eta + i + (2*io+1)*elements, out_im);
        }
      }
    }
  }
#endif
}


#endif
#endif // DIRAC_SSE_H
