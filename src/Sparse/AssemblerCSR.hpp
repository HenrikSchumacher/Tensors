#pragma once

namespace Tensors
{
    namespace Sparse
    {
        template<typename Int_, typename LInt_>
        class AssemblerCSR
        {
        public:
            
            ASSERT_INT(Int_);
            ASSERT_INT(LInt_);
            
            using Int  = Int_;
            using LInt = LInt_;
    
            BinaryMatrix_T = SparseBinaryMatrixCSR<Int,LInt>;
            
        protected:
            
            BinaryMatrix_T pattern;
            BinaryMatrix_T assembler;
            
        public:

            AssemblerCSR() = default;
            
            AssemblerCSR(
                const LInt nnz,
                const Int * const i,
                const Int * const j,
                const Int m,
                const Int n,
                const Int thread_count,
                const bool compressQ  = true,
                const int  symmetrize = 0
            )
            {
                Tensor1<LInt,LInt> from = iota<LInt>( nnz );
                
                Sparse::MatrixCSR<LInt,Int,LInt> A (
                    nnz, i, j, from.data(), m, n, thread_count, false, symmetrize
                );
                
                
                mref<Tensor1<LInt,LInt>> to = A.Values();
                
                // Now we know that from[i] is mapped to nonzero number to[i].
                
                // However, A contains duplicates in the column indices.
                // So we have to do the compression ourselves and track that also in to.
                
                // TODO: Check and debug the compression phase.
                
                A.RequireJobPtr();
                A.SortInner();
                
                Tensor1<LInt,Int> new_outer (A.Outer().Size(),0);
                
                cptr<LInt> A_outer     = A.Outer().data();
                mptr<Int > A_inner     = A.Inner().data();
                mptr<LInt> new_A_outer = new_outer.data();
                
                const auto & job_ptr = A.JobPtr();
                
                ParallelDo(
                    [=,this,&job_ptr]( const Int thread )
                    {
                        const Int i_begin = job_ptr[thread  ];
                        const Int i_end   = job_ptr[thread+1];

                        // To where we write.
                        LInt jj_new        = A_outer[i_begin];
                        LInt next_jj_begin = A_outer[i_begin];
                        
                        for( Int i = i_begin; i < i_end; ++i )
                        {
                            const LInt jj_begin = next_jj_begin;
                            const LInt jj_end   = A_outer[i+1];
                            
                            // Memorize the next entry in outer because outer will be overwritten
                            next_jj_begin = jj_end;
                            
                            LInt row_nonzero_counter = 0;
                            
                            // From where we read.
                            LInt jj = jj_begin;
                            
                            while( jj< jj_end )
                            {
                                const LInt pos = jj;
                                
                                Int j = A_inner[jj];
                                to[jj++] = jj_new;
                                
                                while( (jj < jj_end) && (j == A_inner[jj]) )
                                {
                                    to[jj++] = jj_new;
                                }
                                
                                A_inner[jj_new] = j;
                                
                                ++jj_new;
                                ++row_nonzero_counter;
                            }
                            
                            new_A_outer[i+1] = row_nonzero_counter;
                        }
                    },
                    thread_count
                );
                
                // This is the new array of outer indices.
                new_outer.Accumulate( thread_count );
                
                const LInt compressed_nnz = new_outer[m];
                
                
                // Now we create a new array for new_inner and copy inner to it, eliminating the gaps in between.
                
                Tensor1<Int,LInt> new_inner  (compressed_nnz);
                
                mptr<Int> new_A_inner = new_inner.data();
                
                //TODO: Parallelization might be a bad idea here.
                
                ParallelDo(
                    [=,this]( const Int thread )
                    {
                        const  Int i_begin = job_ptr[thread  ];
                        const  Int i_end   = job_ptr[thread+1];
                        
                        const LInt new_pos = new_A_outer[i_begin];
                        const LInt     pos =     A_outer[i_begin];
                        
                        const LInt thread_nonzeroes = new_A_outer[i_end] - new_A_outer[i_begin];
                        
                        // Starting position of thread in inner list.
                        
                        copy_buffer( &A_inner[pos], &new_A_inner[new_pos], thread_nonzeroes );
                    },
                    thread_count
                );
                
                pattern = BinaryMatrix_T(
                    std::move(new_outer), std::move(new_inner), m, n, thread_count
                );
                
                assember = BinaryMatrix_T(
                    to.data(), from.data(), compress_nnz, nnz, thread_count, compressQ, false
                );
            }
            
            cref<BinaryMatrix_T> PatternMatrix() const
            {
                return pattern;
            }
            
            cref<BinaryMatrix_T> AssemblyMatrix() const
            {
                return assember;
            }
            
            LInt NonzeroCount() const
            {
                return pattern.NonzeroCount();
            }
            
            LInt AssembledNonzeroCount() const
            {
                return assembler.RowCount();
            }
            
            LInt InputNonzeroCount() const
            {
                return assembler.ColCount();
            }
            
            template<typename Scal>
            void AssembleValues( cptr<Scal> input_values, mptr<Scal> assembled_values )
            {
                assembler.Dot( 
                    Scal(1), input_values.data(),     Int(1),
                    Scal(0), assembled_values.data(), Int(1),
                    Int(1)
                );
            }
            
            // Copy constructor
            AssemblerCSR( const MatrixCSR & other ) noexcept
            :   pattern    ( other.pattern )
            ,   assembler  ( other.assembler )
            {}
            
            friend void swap (AssemblerCSR & A, AssemblerCSR & B ) noexcept
            {
                // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
                using std::swap;
                
                swap( A.pattern,   B.pattern );
                swap( A.assembler, B.assembler  );
            }
            
            // (Copy-)assignment operator
            AssemblerCSR & operator=( MatrixCSR other ) noexcept // Pass by value is okay, because we use copy-swap idiom and copy elision.
            {
                // see https://stackoverflow.com/a/3279550/8248900 for details
                
                swap(*this, other);
                
                return *this;
            }
            
            // Move constructor
            AssemblerCSR( MatrixCSR && other ) noexcept
            :   MatrixCSR()
            {
                swap(*this, other);
            }
            

            
            
            virtual ~AssemblerCSR() override = default;
            
        protected:
            
            static std::string ClassName()
            {
                return std::string("Sparse::AssemblerCSR<")+TypeName<Scal>+","+TypeName<Int>+ >";
            }
            
        }; // AssemblerCSR
            
    } // namespace Sparse
    
} // namespace Tensors

