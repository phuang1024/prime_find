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
    /* Free memory */
    cudaFree(_data);
}

MODS Number::Number() {
    /* Initialize with 50 bytes of 0 */
    _init(50);
}

MODS UCH Number::get(const UINT pos) {
    /* Get the byte at pos */
    return _data[pos];
}

MODS void Number::set(const UINT pos, const UCH value) {
    /* Set the byte at pos to value. */
    _resize(pos+1);
    _data[pos] = value;
}


MODS UINT Number::_init(const UINT size) {
    /* Allocate size bytes of 0 */
    cudaMallocManaged(&_data, size);
    cudaMemset(_data, 0, size);
    _size = size;
}

MODS UINT Number::_resize(const UINT target) {
    /* Resize data to target (or more). */
    if (target > _size) {
        const UINT new_size = max(target, _size+50);

        UCH* new_mem;
        cudaMallocManaged(&new_mem, new_size);
        cudaMemcpy(new_mem, _data, _size, cudaMemcpyDeviceToDevice);
        cudaMemset(new_mem+_size, 0, new_size-_size);
        cudaFree(_data);
        _data = new_mem;
        _size = new_size;

        return new_size;
    }
}

MODS void Number::_add(const char* data, const UINT size, const UINT pos) {
    /* Add info from data to internal data starting from pos for size bytes */
    UINT remain = 0;
    for (UINT p = pos; p < pos+size; p++) {
        const UINT new_value = _data[p] + data[p] + remain;
        const UINT byte = new_value & 255;
        remain = new_value >> 8;
        set(pos, byte);
    }
    set(pos+size, _data[pos+size]+remain);
}
