//
//  Prime Find
//  Mersenne prime search using CUDA.
//  Copyright Patrick Huang 2021
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <https://www.gnu.org/licenses/>.
//

#include "utils.cuh"
#include "number.cuh"


MODS Number::~Number() {
    cudaFree(_data);
}

MODS Number::Number() {
    _init(10);
}


MODS UINT Number::_init(const UINT size) {
    cudaMallocManaged(&_data, sizeof(UINT)*size);
    cudaMemset(_data, 0, sizeof(UINT)*size);
    _size = size;
}

MODS UINT Number::_resize(const UINT target) {
    const UINT new_size = max(target, _size*2);

    UINT* new_mem;
    cudaMallocManaged(&new_mem, sizeof(UINT)*new_size);
    cudaMemcpy(new_mem, _data, sizeof(UINT)*_size, cudaMemcpyDeviceToDevice);
    cudaFree(_data);
    _data = new_mem;
    _size = new_size;

    return new_size;
}
