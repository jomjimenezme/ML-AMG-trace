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

#ifndef VCYCLE_double_HEADER
  #define VCYCLE_double_HEADER
  
  #include "interpolation_double.h"
  #include "coarse_oddeven_double.h"
  #include "schwarz_double.h"
  #include "gathering_double.h"
  #include "main_post_def_double.h"
  #include "oddeven_double.h"

  #include "threading.h"
  #include "solver_analysis.h"

  void smoother_double( vector_double phi, vector_double Dphi, vector_double eta,
                           int n, const int res, level_struct *l, struct Thread *threading );
    
  void vcycle_double( vector_double phi, vector_double Dphi, vector_double eta,
                         int res, level_struct *l, struct Thread *threading );
  
#endif
