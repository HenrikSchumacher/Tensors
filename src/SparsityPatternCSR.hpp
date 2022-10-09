#pragma once

namespace Tensors
{
    
    template<typename Int>
    inline void AccumulateAssemblyCounters( Tensor2<Int,Int> & counters )
    {
        ptic("AccumulateAssemblyCounters");

        const Int thread_count = counters.Dimension(0);

        const Int m = counters.Dimension(1);

        for( Int thread = 1; thread < thread_count; ++thread )
        {
            counters(thread, 0) += counters(thread-1, 0);
        }
        for( Int i = 1; i < m; ++i )
        {
            counters(0, i) += counters(thread_count-1, i-1);

            for( Int thread = 1; thread < thread_count; ++thread )
            {
                counters(thread, i) += counters(thread-1, i);
            }
        }

        ptoc("AccumulateAssemblyCounters");
    }
    
    
    template<typename T, typename Int>
    inline void AccumulateAssemblyCounters_Parallel( Tensor2<T,Int> & counters )
    {
        static_assert(CACHE_LINE_WIDTH % sizeof(T) == 0, "CACHE_LINE_WIDTH is not divisible by sizeof(T)");

        constexpr Int per_line = CACHE_LINE_WIDTH / sizeof(T);

        ptic("AccumulateAssemblyCounters (parallel)");
        
        const Int thread_count = counters.Dimension(0);
        
        const Int            m = counters.Dimension(1);
        
        const Int line_count = (m * sizeof(T) + CACHE_LINE_WIDTH - 1 ) / CACHE_LINE_WIDTH;
        
//        valprint("line_count",line_count);
        
        T * S_buffer = nullptr;
        safe_alloc(S_buffer,thread_count+1);
        T * restrict const S = S_buffer;
        S[0] = static_cast<T>(0);

        const Int step = line_count / thread_count;
        const Int corr = line_count % thread_count;
        
//        for( Int thread = 0; thread < thread_count; ++thread )
//        {
//            // each thread does the accumulation on its chunk independently
//            const Int j_begin = (step*(thread  ) + (corr*(thread  ))/thread_count) * per_line;
//            const Int j_end   = std::min(m, (step*(thread+1) + (corr*(thread+1))/thread_count) * per_line);
//
//            print("thread = "+ToString(thread)+", j_begin = "+ToString(j_begin)+", j_end = "+ToString(j_end));
//        }
        
//        tic("local acc");
        #pragma omp parallel for num_threads( thread_count )
        for( Int thread = 0; thread < thread_count; ++thread )
        {
            // each thread does the accumulation on its chunk independently
            const Int j_begin = (step*(thread  ) + (corr*(thread  ))/thread_count) * per_line;
            const Int j_end   = std::min(m, (step*(thread+1) + (corr*(thread+1))/thread_count) * per_line);
            
            if( j_end > j_begin )
            {
                for( Int i = 1; i < thread_count; ++i )
                {
                    counters[i][j_begin] += counters[i-1][j_begin];
                }
                
                for( Int j = j_begin+1; j < j_end; ++j )
                {
                    counters[0][j] += counters[thread_count-1][j-1];
                    
                    for( Int i = 1; i < thread_count; ++i )
                    {
                        counters[i][j] += counters[i-1][j];
                    }
                }
                
                S[thread+1] = counters(thread_count-1,j_end-1);
            }
            else
            {
                S[thread+1] = static_cast<T>(0);
            }
        }
//        toc("local acc");
        
//        for( Int i = 0; i < thread_count; ++i )
//        {
//            valprint("S[i]",S[i]);
//        }
        // scan through the last results of each chunk
        {
            T s_local = static_cast<T>(0);
            for( Int i = 0; i < thread_count; ++i )
            {
                s_local += S[i+1];
                S[i+1] = s_local;
            }
        }

//        for( Int i = 0; i < thread_count; ++i )
//        {
//            valprint("S[i]",S[i]);
//        }
//
//        tic("correction");
        #pragma omp parallel for num_threads( thread_count )
        for( Int thread = 0; thread < thread_count; ++ thread )
        {
            // each thread adds-in its correction
            const T correction = S[thread];
            
            const Int j_begin = (step*(thread  ) + (corr*(thread  ))/thread_count) * per_line;
            const Int j_end   = std::min(m, (step*(thread+1) + (corr*(thread+1))/thread_count) * per_line);

            
            for( Int i = 0; i < thread_count; ++i )
            {
                T * restrict const c_i = counters.data(i);
                
                #pragma omp simd
                for( Int j = j_begin; j < j_end; ++j )
                {
                    c_i[j] += correction;
                }
            }
            
        }
//        toc("correction");
        
        ptoc("AccumulateAssemblyCounters (parallel)");
    }
    
    
    
    
    template<typename Int>
    inline Tensor2<Int,Int> AssemblyCounters(
        const Int * const * const idx,
        const Int * const * const jdx,
        const Int * entry_counts,
        const Int list_count,
        const Int m,
        const int symmetrize = 0
    )
    {
        ptic("AssemblyCounters");
        
        Tensor2<Int,Int> counters (list_count, m, static_cast<Int>(0));

        // https://en.wikipedia.org/wiki/Counting_sort
        // using parallel count sort to sort the cluster (i,j)-pairs according to i.
        // storing counters of each i-index in thread-interleaved format
        // TODO: Improve data layout (transpose counts).
        #pragma omp parallel for num_threads( list_count )
        for( Int thread = 0; thread < list_count; ++thread )
        {
            const Int * restrict const thread_idx = idx[thread];
            const Int * restrict const thread_jdx = jdx[thread];
            
            const Int entry_count = entry_counts[thread];
            
            Int * restrict const c = counters.data(thread);
            
            if( symmetrize!=0 )
            {
                for( Int k = 0; k < entry_count; ++k )
                {
                    const Int i = thread_idx[k];
                    const Int j = thread_jdx[k];
                    
                    c[i] ++;
                    c[j] += static_cast<Int>(i != j);
                }
            }
            else
            {
                for( Int k = 0; k < entry_count; ++k )
                {
                    const Int i = thread_idx[k];
                    
                    ++c[i];
                }
            }
        }
        
//        print(counters.ToString());
//        AccumulateAssemblyCounters(counters);
        AccumulateAssemblyCounters_Parallel(counters);
        
//        print(counters.ToString());
        
        ptoc("AssemblyCounters");
        
        return counters;
    }
    
    template<typename Int> class SparseBinaryMatrixCSR;
    
#define CLASS SparsityPatternCSR
    
    template<typename Int>
    class CLASS
    {
        ASSERT_INT (Int);
        
    protected:
        
        Tensor1<Int,Int> outer;
        Tensor1<Int,Int> inner;
        
        Int m;
        Int n;
        
        Int thread_count = 1;
        
        bool symmetric       = false;
        bool uppertriangular = false;
        bool lowertriangular = false;
        
        // diag_ptr[i] is the first nonzero element in row i such that inner[diag_ptr[i]] >= i
        mutable Tensor1<Int,Int> diag_ptr;
        mutable JobPointers<Int> job_ptr;
        mutable JobPointers<Int> upper_triangular_job_ptr;
        mutable JobPointers<Int> lower_triangular_job_ptr;
        
    public:
        friend class SparseBinaryMatrixCSR<Int>;
        
        CLASS() : m(static_cast<Int>(0)), n(static_cast<Int>(0)) {}

        CLASS(
            const long long m_,
            const long long n_,
            const long long thread_count_
        )
        :   outer       ( Tensor1<Int,Int>(static_cast<Int>(m_+1),static_cast<Int>(0))  )
        ,   m           ( static_cast<Int>(m_)                                    )
        ,   n           ( static_cast<Int>(n_)                                    )
        ,   thread_count( static_cast<Int>(thread_count_)                         )
        {
            Init();
        }
        
        CLASS(
            const long long m_,
            const long long n_,
            const long long nnz_,
            const long long thread_count_
        )
        :   outer       ( Tensor1<Int,Int>(static_cast<Int>(m_+1),static_cast<Int>(0))  )
        ,   inner       ( Tensor1<Int,Int>(static_cast<Int>(nnz_) )                   )
        ,   m           ( static_cast<Int>(m_)                                    )
        ,   n           ( static_cast<Int>(n_)                                    )
        ,   thread_count( static_cast<Int>(thread_count_)                         )
        {
            Init();
        }
        
        
        template<typename J0, typename J1>
        CLASS(
            const J0 * const outer_,
            const J1 * const inner_,
            const long long m_,
            const long long n_,
            const long long thread_count_
        )
        :   outer       ( ToTensor1<Int,Int>(outer_,static_cast<Int>(m_+1))       )
        ,   inner       ( ToTensor1<Int,Int>(inner_,static_cast<Int>(outer_[m_])) )
        ,   m           ( static_cast<Int>(m_)                                )
        ,   n           ( static_cast<Int>(n_)                                )
        ,   thread_count( static_cast<Int>(thread_count_)                     )
        {
            Init();
        }
        
        // Copy constructor
        CLASS( const CLASS & other )
        :   outer       ( other.outer         )
        ,   inner       ( other.inner         )
        ,   m           ( other.m             )
        ,   n           ( other.n             )
        ,   thread_count( other.thread_count  )
        {}
        
        friend void swap (CLASS &A, CLASS &B ) noexcept
        {
            // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
            using std::swap;

            swap( A.outer,        B.outer        );
            swap( A.inner,        B.inner        );
            swap( A.m,            B.m            );
            swap( A.n,            B.n            );
            swap( A.thread_count, B.thread_count );
        }
        
        // Copy assignment operator
        CLASS & operator=(CLASS other)
        {
            // copy-and-swap idiom
            // see https://stackoverflow.com/a/3279550/8248900 for details

            swap(*this, other);

            return *this;
        }

        // Move constructor
        CLASS( CLASS && other ) noexcept : CLASS()
        {
            swap(*this, other);
        }
        
        
        CLASS(
            std::vector<Int> & idx,
            std::vector<Int> & jdx,
            const Int m_,
            const Int n_,
            const Int final_thread_count,
            const bool compress   = true,
            const int  symmetrize = 0
        )
        :   CLASS ( m_, n_, static_cast<Int>(1) )
        {
            const Int * i = idx.data();
            const Int * j = jdx.data();
            
            Tensor1<Int,Int> entry_counts (1, static_cast<Int>(idx.size()));
                        
            FromPairs( &i, &j, entry_counts.data(),
                    1, final_thread_count, compress, symmetrize );
        }
        
        CLASS(
            const std::vector<std::vector<Int>> & idx,
            const std::vector<std::vector<Int>> & jdx,
            const Int m_,
            const Int n_,
            const Int final_thread_count,
            const bool compress   = true,
            const int  symmetrize = 0
        )
        : CLASS ( m_, n_, static_cast<Int>(idx.size()) )
        {
            Int list_count = static_cast<Int>(idx.size());
            Tensor1<const Int*, Int> i (list_count);
            Tensor1<const Int*, Int> j (list_count);

            Tensor1<Int,Int> entry_counts (list_count);
            
            for( Int thread = 0; thread < list_count; ++thread )
            {
                i[thread] = idx[thread].data();
                j[thread] = jdx[thread].data();
                entry_counts[thread] = static_cast<Int>(idx[thread].size());
            }
            
            FromPairs( i.data(), j.data(), entry_counts.data(),
                    list_count, final_thread_count, compress, symmetrize );
        }
    
        virtual ~CLASS() = default;
        

    public:
        
        Int ThreadCount() const
        {
            return thread_count;
        }
        
        void SetThreadCount( const Int thread_count_ )
        {
            thread_count = std::max( static_cast<Int>(1), thread_count_);
        }
        
    protected:
        
        void Init()
        {
            outer[0] = static_cast<Int>(0);
        }
        
        void FromPairs(
            const Int * const * const idx,
            const Int * const * const jdx,
            const Int * entry_counts,
            const Int list_count,
            const Int final_thread_count,
            const bool compress   = true,
            const int  symmetrize = 0
        )
        {
            ptic(ClassName()+"::FromPairs");
            
            Tensor2<Int,Int> counters = AssemblyCounters(
                idx, jdx, entry_counts, list_count, m, symmetrize
            );
            
            const Int nnz = counters(list_count-1,m-1);
            
            if( nnz > 0 )
            {
                inner = Tensor1<Int,Int>( nnz );
            
                Int * restrict const outer__ = outer.data();
                Int * restrict const inner__ = inner.data();

                copy_buffer( counters.data(list_count-1), &outer__[1], m );
                
                // writing the j-indices into sep_column_indices
                // the counters array tells each thread where to write
                // since we have to decrement entries of counters array, we have to loop in reverse order to make the sort stable in the j-indices.
                
                if( symmetrize != 0 )
                {
                    #pragma omp parallel for num_threads( list_count )
                    for( Int thread = 0; thread < list_count; ++thread )
                    {
                        const Int entry_count = entry_counts[thread];
                        
                        const Int * restrict const thread_idx = idx[thread];
                        const Int * restrict const thread_jdx = jdx[thread];
                        
                              Int * restrict const c = counters.data(thread);
                        
                        for( Int k = entry_count - 1; k > -1; --k )
                        {
                            const Int i = thread_idx[k];
                            const Int j = thread_jdx[k];
                            {
                                const Int pos  = --c[i];
                                inner__[pos] = j;
                            }
                            
                            c[j] -= static_cast<Int>(i != j);
                            
                            const Int pos  = c[j];
                            
                            inner__[pos] = i;
                        }
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads( list_count )
                    for( Int thread = 0; thread < list_count; ++thread )
                    {
                        const Int entry_count = entry_counts[thread];
                        
                        const Int * restrict const thread_idx = idx[thread];
                        const Int * restrict const thread_jdx = jdx[thread];
                        
                              Int * restrict const c = counters.data(thread);
                        
                        for( Int k = entry_count - 1; k > -1; --k )
                        {
                            const Int i = thread_idx[k];
                            const Int j = thread_jdx[k];
                            {
                                const Int pos  = --c[i];
                                inner__[pos] = j;
                            }
                        }
                    }
                }

                // From here on, we may use as many threads as we want.
                SetThreadCount( final_thread_count );

                if( compress )
                {
                    Compress();
                }

                // We have to sort b_inner to be compatible with the CSR format.
                SortInner();
            }
            else
            {
                SetThreadCount( final_thread_count );
            }
            ptoc(ClassName()+"::FromPairs");
        }
        
        void RequireJobPtr() const
        {
            if( job_ptr.Size()-1 != thread_count )
            {
                ptic(ClassName()+"::RequireJobPtr");
                
//                // TODO: Find better cost model.
//                Tensor1<Int,Int> costs ( outer.data(), m+1 );
//
//                for( Int i = 0; i < m ; ++i )
//                {
//                    costs[i+1] += i;
//                }
//
//                job_ptr = JobPointers<Int>( m, costs.data(), thread_count, false );
                
                job_ptr = JobPointers<Int>( m, outer.data(), thread_count, false );
                
                ptoc(ClassName()+"::RequireJobPtr");
            }
        }
        
        void RequireDiag() const
        {
            if( diag_ptr.Size() != m )
            {
                ptic(ClassName()+"::RequireDiag");
                
                if( outer.Last() <= 0 )
                {
                    diag_ptr = Tensor1<Int,Int>( outer.data()+1, m );
                }
                else
                {
                    RequireJobPtr();
                    
                    diag_ptr = Tensor1<Int,Int>( m );
                    
                          Int * restrict const diag_ptr__ = diag_ptr.data();
                    const Int * restrict const outer__    = outer.data();
                    const Int * restrict const inner__    = inner.data();

                    #pragma omp parallel for num_threads( thread_count )
                    for( Int thread = 0; thread < thread_count; ++thread )
                    {
                        const Int i_begin = job_ptr[thread  ];
                        const Int i_end   = job_ptr[thread+1];

                        for( Int i = i_begin; i < i_end; ++ i )
                        {
                            const Int k_begin = outer__[i  ];
                            const Int k_end   = outer__[i+1];

                            Int k = k_begin;

                            while( (k < k_end) && (inner__[k] < i)  )
                            {
                                ++k;
                            }
                            diag_ptr__[i] = k;
                        }
                    }
                }
                
                ptoc(ClassName()+"::RequireDiag");
            }
        }
        
        void RequireUpperTriangularJobPtr() const
        {
            if( (m > 0) && (upper_triangular_job_ptr.Size()-1 != thread_count) )
            {
                ptic(ClassName()+"::RequireUpperTriangularJobPtr");
                
                RequireDiag();
                
                Tensor1<Int,Int> costs = Tensor1<Int,Int>(m + 1);
                costs[0]=0;
                
                const Int * restrict const diag_ptr__ = diag_ptr.data();
                const Int * restrict const outer__    = outer.data();
                      Int * restrict const costs__    = costs.data();
                
                #pragma omp parallel for num_threads( thread_count )
                for( Int thread = 0; thread < thread_count; ++thread )
                {
                    const Int i_begin = job_ptr[thread  ];
                    const Int i_end   = job_ptr[thread+1];
                    
                    for( Int i = i_begin; i < i_end; ++ i )
                    {
                        costs__[i+1] = outer__[i+1] - diag_ptr__[i];
                    }
                }
                
                costs.Accumulate( thread_count );
                
                upper_triangular_job_ptr = JobPointers( m, costs.data(), thread_count, false );
                
                ptoc(ClassName()+"::RequireUpperTriangularJobPtr");
            }
        }
        
        void RequireLowerTriangularJobPtr() const
        {
            if( (m > 0) && (lower_triangular_job_ptr.Size()-1 != thread_count) )
            {
                ptic(ClassName()+"::RequireLowerTriangularJobPtr");
                
                RequireDiag();
                                
                Tensor1<Int,Int> costs = Tensor1<Int,Int>(m + 1);
                costs[0]=0;
                
                const Int * restrict const diag_ptr__ = diag_ptr.data();
                const Int * restrict const outer__    = outer.data();
                      Int * restrict const costs__    = costs.data();
                
                #pragma omp parallel for num_threads( thread_count )
                for( Int thread = 0; thread < thread_count; ++thread )
                {
                    
                    const Int i_begin = job_ptr[thread  ];
                    const Int i_end   = job_ptr[thread+1];
                    
                    for( Int i = i_begin; i < i_end; ++ i )
                    {
                        costs__[i+1] = diag_ptr__[i] - outer__[i];
                    }
                }
                
                costs.Accumulate( thread_count );
                
                lower_triangular_job_ptr = JobPointers( m, costs.data(), thread_count, false );
                
                ptoc(ClassName()+"::RequireLowerTriangularJobPtr");
            }
        }
        
    public:

        Int RowCount() const
        {
            return m;
        }
        
        Int ColCount() const
        {
            return n;
        }
        
        Int NonzeroCount() const
        {
            return inner.Size();
        }

        Tensor1<Int,Int> & Outer()
        {
            return outer;
        }
        
        const Tensor1<Int,Int> & Outer() const
        {
            return outer;
        }

        Tensor1<Int,Int> & Inner()
        {
            return inner;
        }
        
        const Tensor1<Int,Int> & Inner() const
        {
            return inner;
        }
        
        const Tensor1<Int,Int> & Diag() const
        {
            RequireDiag();
            
            return diag_ptr;
        }
        
        
        const JobPointers<Int> & JobPtr() const
        {
            RequireJobPtr();
            
            return job_ptr;
        }
        
        
        
        const JobPointers<Int> & UpperTriangularJobPtr() const
        {
            RequireUpperTriangularJobPtr();
            
            return upper_triangular_job_ptr;
        }
        
        
        const JobPointers<Int> & LowerTriangularJobPtr() const
        {
            RequireLowerTriangularJobPtr();
            
            return lower_triangular_job_ptr;
        }
        
        
    protected:
        
        Tensor2<Int,Int> CreateTransposeCounters() const
        {
            ptic(ClassName()+"::CreateTransposeCounters");
            
            RequireJobPtr();
            
            Tensor2<Int,Int> counters ( thread_count, n, static_cast<Int>(0) );
            
            if( WellFormed() )
            {
    //            ptic("Counting sort");
                // Use counting sort to sort outer indices of output matrix.
                // https://en.wikipedia.org/wiki/Counting_sort
    //            ptic("Counting");
                
                
                #pragma omp parallel for num_threads( thread_count )
                for( Int thread = 0; thread < thread_count; ++thread )
                {
                    const Int i_begin = job_ptr[thread  ];
                    const Int i_end   = job_ptr[thread+1];
                    
                          Int * restrict const c = counters.data(thread);
                    
                    const Int * restrict const A_outer  = Outer().data();
                    const Int * restrict const A_inner  = Inner().data();
                                    
                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        const Int jj_begin = A_outer[i  ];
                        const Int jj_end   = A_outer[i+1];
                        
                        for( Int jj = jj_begin; jj < jj_end; ++jj )
                        {
                            const Int j = A_inner[jj];
                            ++c[j];
                        }
                    }
                }

                AccumulateAssemblyCounters_Parallel( counters );
            }
            
            ptoc(ClassName()+"::CreateTransposeCounters");
            
            return counters;
        }
        
    public:
        
        virtual void SortInner()
        {
            // Sorts the column indices of each matrix row.
            
            ptic(ClassName()+"::SortInner");
            
//            print("TimSort");
            
            if( WellFormed() )
            {
                RequireJobPtr();
                
                #pragma omp parallel for num_threads( thread_count )
                for( Int thread = 0; thread < thread_count; ++thread )
                {
//                    TimSort<Int,Int> tim_sort(512);
                    
                    const Int i_begin = job_ptr[thread  ];
                    const Int i_end   = job_ptr[thread+1];
                
                    const Int * restrict const rp = outer.data();
                          Int * restrict const ci = inner.data();
                    
                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        const Int begin = rp[i  ];
                        const Int end   = rp[i+1];
                        
                        std::sort( &ci[begin], &ci[end] );
//                        tim_sort( &ci[begin], &ci[end] );
                    }
                }
            }
            
            ptoc(ClassName()+"::SortInner");
        }
        
    public:
        
        
        virtual void Compress()
        {
            ptic(ClassName()+"::Compress");
            
            if( WellFormed() )
            {
                RequireJobPtr();
                
                Tensor1<Int,Int> new_outer (outer.Size(),0);
                
                Int * restrict const new_outer__ = new_outer.data();
                Int * restrict const     outer__ = outer.data();
                Int * restrict const     inner__ = inner.data();
                
                #pragma omp parallel for num_threads( thread_count )
                for( Int thread = 0; thread < thread_count; ++thread )
                {
                    
                    const Int i_begin = job_ptr[thread  ];
                    const Int i_end   = job_ptr[thread+1];
                    
                    // To where we write.
                    Int jj_new        = outer__[i_begin];
                    
                    // Memoize the next entry in outer because outer will be overwritten
                    Int next_jj_begin = outer__[i_begin];
                    
                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        const Int jj_begin = next_jj_begin;
                        const Int jj_end   = outer__[i+1];
                        
                        // Memoize the next entry in outer because outer will be overwritten
                        next_jj_begin = jj_end;
                        
                        Int row_nonzero_counter = static_cast<Int>(0);
                        
                        // From where we read.
                        Int jj = jj_begin;
                        
                        while( jj< jj_end )
                        {
                            Int j = inner__[jj];
                            
                            {
                                if( jj > jj_new )
                                {
                                    inner__[jj] = static_cast<Int>(0);
                                }
                                
                                ++jj;
                            }
            
                            while( (jj < jj_end) && (j == inner__[jj]) )
                            {
                                if( jj > jj_new )
                                {
                                    inner__[jj] = static_cast<Int>(0);
                                }
                                ++jj;
                            }
                            
                            inner__[jj_new] = j;
                            
                            jj_new++;
                            row_nonzero_counter++;
                        }
                        new_outer__[i+1] = row_nonzero_counter;
                    }
                }
                
                // This is the new array of outer indices.
                new_outer.Accumulate( thread_count  );
                
                const Int nnz = new_outer[m];
                
                Tensor1<Int,Int> new_inner (nnz,0);
                
                //TODO: Parallelization might be a bad idea here.
                
                #pragma omp parallel for num_threads( thread_count )
                for( Int thread = 0; thread < thread_count; ++thread )
                {
                    const Int i_begin = job_ptr[thread  ];
                    const Int i_end   = job_ptr[thread+1];
                    
                    const Int new_pos = new_outer__[i_begin];
                    const Int     pos =     outer__[i_begin];

                    const Int thread_nonzeroes = new_outer__[i_end] - new_outer__[i_begin];
                    
                    copy_buffer( &inner.data()[pos], &new_inner.data()[new_pos], thread_nonzeroes );
                }
                
                swap( new_outer,  outer  );
                swap( new_inner,  inner  );
                
                job_ptr = JobPointers<Int>();
            }
            
            ptoc(ClassName()+"::Compress");
        }
        
//##############################################################################################
//####          Matrix Multiplication
//##############################################################################################
        
    public:
        
        CLASS<Int> DotBinary_( const CLASS<Int> & B ) const
        {
            ptic(ClassName()+"::DotBinary_");
                        
            if( WellFormed() && B.WellFormed() )
            {
                RequireJobPtr();
                
                ptic("Create counters for counting sort");
                
                Tensor2<Int,Int> counters ( thread_count, m, static_cast<Int>(0) );
                
                // Expansion phase, utilizing counting sort to generate expanded row pointers and column indices.
                // https://en.wikipedia.org/wiki/Counting_sort
                #pragma omp parallel for num_threads( thread_count )
                for( Int thread = 0; thread < thread_count; ++thread )
                {
                    const Int i_begin = job_ptr[thread  ];
                    const Int i_end   = job_ptr[thread+1];
                    
                          Int * restrict const c = counters.data(thread);
                    
                    const Int * restrict const A_outer  = Outer().data();
                    const Int * restrict const A_inner  = Inner().data();
                    
                    const Int * restrict const B_outer  = B.Outer().data();
                    
                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        const Int jj_begin = A_outer[i  ];
                        const Int jj_end   = A_outer[i+1];
                        
                        for( Int jj = jj_begin; jj < jj_end; ++jj )
                        {
                            const Int j = A_inner[jj];
                            
                            c[i] += (B_outer[j+1] - B_outer[j]);
                        }
                    }
                }
                
                ptoc("Create counters for counting sort");
                
                AccumulateAssemblyCounters_Parallel(counters);
                
                const Int nnz = counters.data(thread_count-1)[m-1];
                
                CLASS<Int> C ( m, B.ColCount(), nnz, thread_count );
                
                copy_buffer( counters.data(thread_count-1), &C.Outer().data()[1], m );
                
                ptic("Counting sort");
                
                #pragma omp parallel for num_threads( thread_count )
                for( Int thread = 0; thread < thread_count; ++thread )
                {
      
                    const Int i_begin = job_ptr[thread  ];
                    const Int i_end   = job_ptr[thread+1];
                    
                          Int * restrict const c = counters.data(thread);

                    const Int * restrict const A_outer  = Outer().data();
                    const Int * restrict const A_inner  = Inner().data();
    //                const T * restrict const A_values = Value().data();
                    
                    const Int * restrict const B_outer  = B.Outer().data();
                    const Int * restrict const B_inner  = B.Inner().data();
    //                const T * restrict const B_values = B.Value().data();
                    
                          Int * restrict const C_inner  = C.Inner().data();
    //                      T * restrict const C_values = C.Value().data();
                    
                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        const Int jj_begin = A_outer[i  ];
                        const Int jj_end   = A_outer[i+1];
                        
                        for( Int jj = jj_begin; jj < jj_end; ++jj )
                        {
                            const Int j = A_inner[jj];
                            
                            const Int kk_begin = B_outer[j  ];
                            const Int kk_end   = B_outer[j+1];
                            
                            for( Int kk = kk_end-1; kk > kk_begin-1; --kk )
                            {
                                const Int k = B_inner[kk];
                                const Int pos = --c[ i ];
                                
                                C_inner [pos] = k;
    //                            C_values[pos] = A_values[jj] * B_values[kk];
                            }
                        }
                    }
                }
                // Finished expansion phase (counting sort).
                ptoc("Counting sort");
                
                // Now we have to care about the correct ordering of inner indices and values.
                C.SortInner();
                
                // Finally we compress duplicates in inner and values.
                C.Compress();
                
                ptoc(ClassName()+"::DotBinary_");
                
                return C;
            }
            else
            {
                return CLASS<Int> ();
            }
        }
        
    protected:
        
        // Assume all nonzeros are equal to 1.
        template<typename T_ext, typename T_in, typename T_out>
        void Dot_(
            const T_ext alpha,
            const T_in  * X,
            const T_out beta,
                  T_out * Y,
            const Int cols = static_cast<Int>(1)
        ) const
        {
            if( WellFormed() )
            {
                SparseBLAS<T_ext,Int,T_in,T_out> sblas ( thread_count );
                
                sblas.Multiply_BinaryMatrix_DenseMatrix(
                    outer.data(),inner.data(),m,n,alpha,X,beta,Y,cols,JobPtr()
                );
            }
            else
            {
                wprint(ClassName()+"::Dot_: Not WellFormed(). Doing nothing.");
            }
        }
        
        // Supply an external list of values.
        template<typename T_ext, typename T_in, typename T_out>
        void Dot_(
            const T_ext * values,
            const T_ext alpha,
            const T_in  * X,
            const T_out beta,
                  T_out * Y,
            const Int cols = static_cast<Int>(1)
        ) const
        {
            if( WellFormed() )
            {
                auto sblas = SparseBLAS<T_ext,Int,T_in,T_out>( thread_count );
                
                sblas.Multiply_GeneralMatrix_DenseMatrix(
                    outer.data(),inner.data(),values,m,n,alpha,X,beta,Y,cols,JobPtr()
                );
            }
            else
            {
                wprint(ClassName()+"::Dot_: Not WellFormed(). Doing nothing.");
            }
        }
        
        

//##############################################################################################
//####          Lookup Operations
//##############################################################################################
        
        
    private:
        
        void BoundCheck( const Int i, const Int j ) const
        {
            if( (i < 0) || (i > m) )
            {
                eprint(ClassName()+": Row index " + std::to_string(i) + " is out of bounds { 0, " + std::to_string(m-1) +" }.");
            }
            if( (j < 0) || (j > n) )
            {
                eprint(ClassName()+": Column index " + std::to_string(j) + " is out of bounds { 0, " + std::to_string(n-1) +" }.");
            }
        }
        
    public:
        
        Int FindNonzeroPosition( const Int i, const Int j ) const
        {
            // Looks up the entry {i,j}. If existent, its index within the list of nonzeroes is returned. Otherwise, a negative number is returned (-1 if simply not found and -2 if i is out of bounds).
            
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i,j);
#endif
            
            if( (0 <= i) && (i<m) )
            {
                const Int * restrict const inner__ = inner.data();

                Int L = outer[i  ];
                Int R = outer[i+1]-1;
                while( L < R )
                {
                    const Int k = R - (R-L)/static_cast<Int>(2);
                    const Int col = inner__[k];

                    if( col > j )
                    {
                        R = k-1;
                    }
                    else
                    {
                        L = k;
                    }
                }
                return (inner__[L]==j) ? L : static_cast<Int>(-1);
            }
            else
            {
                wprint(ClassName()+"::FindNonzeroPosition: Row index i = "+ToString(i)+" is out of bounds {0,"+ToString(m)+"}.");
                return static_cast<Int>(-2);
            }
        }
        
        
        template<typename S, typename T, typename J>
        void FillLowerTriangleFromUpperTriangle( std::map<S,Tensor1<T,J>> & values ) const
        {
            ptic(ClassName()+"::FillLowerTriangleFromUpperTriangle");
            
            if( WellFormed() )
            {
                const Int * restrict const diag__   = Diag().data();
                const Int * restrict const outer__  = Outer().data();
                const Int * restrict const inner__  = Inner().data();
                
                auto & job_ptr__ = LowerTriangularJobPtr();
                
                #pragma omp parallel for num_threads( thread_count )
                for( Int thread = 0; thread < thread_count; ++thread )
                {
                    
                    const Int i_begin = job_ptr__[thread];
                    const Int i_end   = job_ptr__[thread+1];
                    
                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        const Int k_begin = outer__[i];
                        const Int k_end   =  diag__[i];
                        
                        for( Int k = k_begin; k < k_end; ++k )
                        {
                            const Int j = inner__[k];
                            
                            Int L =  diag__[j];
                            Int R = outer__[j+1]-1;
                            
                            while( L < R )
                            {
                                const Int M   = R - (R-L)/static_cast<Int>(2);
                                const Int col = inner__[M];

                                if( col > i )
                                {
                                    R = M-1;
                                }
                                else
                                {
                                    L = M;
                                }
                            }
                            
                            for( auto & f : values )
                            {
                                f.second[k] = f.second[L];
                            }
                            
                        } // for( Int k = k_begin; k < k_end; ++k )

                    } // for( Int i = i_begin; i < i_end; ++i )
                    
                } // #pragma omp parallel
            }
            
            ptoc(ClassName()+"::FillLowerTriangleFromUpperTriangle");
        }
        
        bool WellFormed() const
        {
            return ( ( outer.Size() > 1 ) && ( outer.Last() > 0 ) );
        }
        
    public:
        
        virtual Int Dimension( const bool dim )
        {
            return dim ? n : m;
        }
        
        std::string Stats() const
        {
            std::stringstream s;
            
            s
            << "\n==== "+ClassName()+" Stats ====" << "\n\n"
            << " RowCount()      = " << RowCount() << "\n"
            << " ColCount()      = " << ColCount() << "\n"
            << " NonzeroCount()  = " << NonzeroCount() << "\n"
            << " ThreadCount()   = " << ThreadCount() << "\n"
            << " Outer().Size()  = " << Outer().Size() << "\n"
            << " Inner().Size()  = " << Inner().Size() << "\n"
            << "\n==== "+ClassName()+" Stats ====\n" << std::endl;
            
            return s.str();
        }
        
        static std::string ClassName()
        {
            return TO_STD_STRING(CLASS)+"<"+TypeName<Int>::Get()+">";
        }
        
    }; // CLASS

    
} // namespace Tensors


#undef CLASS

