#pragma once

namespace Tensors
{
    namespace LAPACK
    {
        template<
            Layout layout, Op op_A,
            typename Scal, typename I0, typename I1, typename I2, typename I3
        >
        TOOLS_FORCE_INLINE Int getrs(
            const I0 n_, const I1 nrhs_,
            Scal * A_, const I2 ldA_,
            Int * perm,
            Scal * B_, const I3 ldB_
        )
        {
            //TODO: !!!
        }
        
    } // namespace LAPACK
    
} // namespace Tensors




