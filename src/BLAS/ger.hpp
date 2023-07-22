#pragma once

namespace Tensors
{
    namespace BLAS
    {
        template<
            Layout layout, Op opx, Op opy, typename Scal,
            typename I0, typename I1, typename I2, typename I3, typename I4
        >
        force_inline void ger(
            const I0 m_, const I1 n_,
            cref<Scal> alpha, cptr<Scal> x, const I2 inc_x_,
                              cptr<Scal> y, const I3 inc_y_,
                              mptr<Scal> A, const I4 ldA_
        )
        {
            // A := alpha * opx(x) * opx(y^T) + A,
            //
            // where {opx,opy} can be {id,id}, {conj,id}, or {id, conj}.
            // Note that {conj, conj} is _not_ allowed (cblas does not allow us to contruct it.
            
            ASSERT_INT(I0);
            ASSERT_INT(I1);
            ASSERT_INT(I2);
            ASSERT_INT(I3);
            ASSERT_INT(I4);
            
            int m     = int_cast<int>(m_);
            int n     = int_cast<int>(n_);
            int ldA   = int_cast<int>(ldA_);
            int inc_x = int_cast<int>(inc_x_);
            int inc_y = int_cast<int>(inc_y_);
            
            assert_positive(m);
            assert_positive(n);
            assert_positive(ldA);
            assert_positive(inc_x);
            assert_positive(inc_y);


            if constexpr ( SameQ<Scal,double> )
            {
                return cblas_dger( to_BLAS(layout), m, n, alpha, x, inc_x, y, inc_y, A, ldA );
            }
            else if constexpr ( SameQ<Scal,float> )
            {
                return cblas_sger( to_BLAS(layout), m, n, alpha, x, inc_x, y, inc_y, A, ldA );
            }
            else if constexpr ( SameQ<Scal,std::complex<double>> )
            {
                if constexpr ( (opx == Op::Id) && ((opy == Op::Id)) )
                {
                    return cblas_zgeru(
                        (layout == Layout::RowMajor) ? Layout::ColMajor : Layout::RowMajor,
                        m, n, &alpha, x, inc_x, y, inc_y, A, ldA
                    );
                }
                else if constexpr ( (opx == Op::Conj) && ((opy == Op::Id)) )
                {
                    return cblas_zgerc(
                        (layout == Layout::RowMajor) ? Layout::ColMajor : Layout::RowMajor,
                        m, n, &alpha, x, inc_x, y, inc_y, A, ldA
                    );
                }
                else if constexpr ( (opx == Op::Id) && ((opy == Op::Conj)) )
                {
                    return cblas_zgerc(
                        to_BLAS(layout),
                        m, n, &alpha, x, inc_x, y, inc_y, A, ldA
                    );
                }
                else
                {
                    eprint("Get does not us to conjugate both input vectors.");
                }
    
            }
            else if constexpr ( SameQ<Scal,std::complex<float>> )
            {
                if constexpr ( (opx == Op::Id) && ((opy == Op::Id)) )
                {
                    return cblas_cgeru(
                        (layout == Layout::RowMajor) ? Layout::ColMajor : Layout::RowMajor,
                        m, n, &alpha, x, inc_x, y, inc_y, A, ldA
                    );
                }
                else if constexpr ( (opx == Op::Conj) && ((opy == Op::Id)) )
                {
                    return cblas_cgerc(
                        (layout == Layout::RowMajor) ? Layout::ColMajor : Layout::RowMajor,
                        m, n, &alpha, x, inc_x, y, inc_y, A, ldA
                    );
                }
                else if constexpr ( (opx == Op::Id) && ((opy == Op::Conj)) )
                {
                    return cblas_cgerc(
                        to_BLAS(layout),
                        m, n, &alpha, x, inc_x, y, inc_y, A, ldA
                    );
                }
            }
            else
            {
                eprint("ger not defined for scalar type " + TypeName<Scal> );
            }
            
        }
        
    } // namespace BLAS
    
} // namespace Tensors

