# Copyright (c) 2018 UT-BATTELLE, LLC
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


crossmatch_mahalanobis <- function(indir1, indir2, caseset1, caseset2, season, ens_set1, ens_set2) {

	suppressMessages(library(ncdf4))

	n_ens1 = length(ens_set1)
	n_ens2 = length(ens_set2)

	total_n = n_ens1 + n_ens2

	template_yr = ens_set1[1]

	file_path = paste0(indir1, "/", caseset1, ".pp/", caseset1, "_", sprintf("%04d", template_yr),".gavg_", season, ".nc")


	#print(file_path)

	iid = nc_open(file_path)
	vars = names(iid[['var']])
	nvars = length(vars)

	#print(nvars)
	#print(vars)

	nc_close(iid)

	X = matrix(data = NA, nrow = total_n, ncol = nvars-2)

	#print(dim(X))


	for (n in 1:n_ens1){
		ens_no = ens_set1[n]
		
		file_path = paste0(indir1, "/", caseset1, ".pp/", caseset1, "_", sprintf("%04d", ens_no), ".gavg_", season, ".nc") 


		iid = nc_open(file_path)

		for (k in 3:nvars){ 
			x_temp = ncvar_get(iid,vars[k])
			X[n, k-2] = x_temp
		}

		nc_close(iid)
	}

	for (n in 1:n_ens2){
		ens_no = ens_set2[n]  

		file_path = paste0(indir2, "/", caseset2, ".pp/", caseset2, "_", sprintf("%04d", ens_no), ".gavg_", season, ".nc")

		iid = nc_open(file_path)

		for (k in 3:nvars){
			x_temp = ncvar_get(iid,vars[k])
			X[n+n_ens1, k-2] = x_temp
		}

		nc_close(iid)
	}


	#print(X)
	#print(dim(X))

	z = c(rep(0,n_ens1),rep(1,n_ens2))
	#print(z)

	## Rank based Mahalanobis distance between each pair:

	n <- dim(X)[1]
	k <- dim(X)[2]

	for (j in 1:k) X[,j] <- rank(X[,j])

	#print(X)

	cv <- cov(X)

	vuntied <- var(1:n)

	rat <- sqrt(vuntied/diag(cv))
	cv <- diag(rat)%*%cv%*%diag(rat)
	out <- matrix(NA,n,n)

	suppressMessages(library(MASS))
	suppressMessages(library(crossmatch))

	icov <- ginv(cv)
	for (i in 1:n) out[i,] <- mahalanobis(X,X[i,],icov,inverted=TRUE)
	dis <- out

	## The cross-match test:
	pdf = crossmatchdist(n_ens1+n_ens2,n_ens1)

	#print(pdf)

	a = crossmatchtest(z,dis)

	#print(a)

	return(a)

}
