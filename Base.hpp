#pragma once

#include <iostream>
#include <random>
#include <cstring>
#include <complex>

#include "submodules/Tools/Tools.hpp"

namespace Tensors
{
    using namespace Tools;
    
    static constexpr Size_T DefaultAlignment = Tools::Alignment;
    
}

#include "src/Tensor1.hpp"
#include "src/Tensor2.hpp"
#include "src/Tensor3.hpp"
