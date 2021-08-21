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


/* Unbounded unsigned integer for CUDA. */
class Number {
public:
    MODS ~Number();
    MODS Number();

    MODS UCH get(const UINT pos);
    MODS void set(const UINT pos, const UCH value);

private:
    MODS UINT _init(const UINT size);
    MODS UINT _resize(const UINT target);

    MODS void _add(const char* data, const UINT size, const UINT pos);

    UCH* _data;
    UINT _size;
};
