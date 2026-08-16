#pragma once
#include <vector>
#include <cstdint>
#include <cassert>
#include <type_traits>


template<typename Tp>
struct BaseMatrix {
        using value_type = Tp;
        using size_type  = uint32_t;
        using reference  = Tp&;
        using pointer    = Tp*;
        using const_reference  = const Tp&;
        using const_pointer    = const Tp*;

    public:
        BaseMatrix() : rows(1), cols(1) {resize(1, 1);}
        BaseMatrix(size_type _rows, size_type _cols)
        : rows(_rows), cols(_cols) {resize(_rows, _cols);}

        constexpr BaseMatrix& operator=(const std::vector<value_type>& _values) noexcept {
            assert(values.size() == _values.size());
            values = _values;
            return *this;
        }

        constexpr BaseMatrix operator+(const BaseMatrix& other) const noexcept {
            assert(isSameShape(other));
            
            BaseMatrix result = BaseMatrix(rows, cols);

            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < cols; ++j)
                    result(i, j) = (*this)(i, j) + other(i, j);

            return result;
        }

        constexpr BaseMatrix operator-(const BaseMatrix& other) const noexcept {
            assert(isSameShape(other));
            
            BaseMatrix result = BaseMatrix(rows, cols);

            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < cols; ++j)
                    result(i, j) = (*this)(i, j) - other(i, j);

            return result;
        }

        // Component-wise multiplication
        constexpr BaseMatrix operator*(const BaseMatrix& other) const noexcept {
            BaseMatrix result = BaseMatrix(rows, cols);

            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < cols; ++j)
                    result(i, j) = (*this)(i, j) * other(i, j);

            return result;
        }

        // Component-wise dividing
        constexpr BaseMatrix operator/(const BaseMatrix& other) const noexcept {
            BaseMatrix result = BaseMatrix(rows, cols);

            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < cols; ++j)
                    result(i, j) = (*this)(i, j) / other(i, j);

            return result;
        }

        constexpr BaseMatrix operator*(float factor) const noexcept {
            BaseMatrix result = BaseMatrix(rows, cols);

            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < cols; ++j)
                    result(i, j) = (*this)(i, j) * factor;

            return result;
        }

        constexpr BaseMatrix operator/(float factor) const noexcept {
            BaseMatrix result = BaseMatrix(rows, cols);

            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < cols; ++j)
                    result(i, j) = (*this)(i, j) / factor;

            return result;
        }

        constexpr BaseMatrix& operator+=(const BaseMatrix& other) noexcept {
            (*this) = (*this) + other;
            return *this;
        }

        constexpr BaseMatrix& operator-=(const BaseMatrix& other) noexcept {
            (*this) = (*this) - other;
            return *this;
        }

        constexpr BaseMatrix& operator*=(const BaseMatrix& other) noexcept {
            (*this) = (*this) * other;
            return *this;
        }

        constexpr BaseMatrix& operator/=(const BaseMatrix& other) noexcept {
            (*this) = (*this) / other;
            return *this;
        }

        constexpr BaseMatrix& operator*=(float factor) noexcept {
            (*this) = (*this) * factor;
            return *this;
        }

        constexpr BaseMatrix& operator/=(float factor) noexcept {
            (*this) = (*this) / factor;
            return *this;
        }

        constexpr bool isSameShape(const BaseMatrix& other) const noexcept {
            return (rows == other.rows) && (cols == other.cols);
        }

        constexpr BaseMatrix transposed() const noexcept {
            BaseMatrix result = BaseMatrix(cols, rows);

            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < cols; ++j)
                    result(j, i) = (*this)(i, j);

            return result;
        }

        constexpr reference operator()(size_type _row)                 noexcept {return values[getIndex(_row, 0)];}
        constexpr reference operator()(size_type _row, size_type _col) noexcept {return values[getIndex(_row, _col)];}
        constexpr const_reference operator()(size_type _row)                 const noexcept {return values[getIndex(_row, 0)];}
        constexpr const_reference operator()(size_type _row, size_type _col) const noexcept {return values[getIndex(_row, _col)];}

        void resize(size_type _rows, size_type _cols) {
            rows = _rows;
            cols = _cols;
            values.assign(rows * cols, static_cast<value_type>(0));
        }

        template<typename F>
        void forEach(F&& f) noexcept {
            constexpr bool withoutRowCol = std::is_invocable_r_v<void, F, float&>;
            
            assert((std::is_invocable_r_v<void, F, float&, size_type, size_type> || std::is_invocable_r_v<void, F, float&>));

            if constexpr (withoutRowCol) {
                for (float& v : values) f(v);
            } else {
                for (size_t i = 0; i < rows; ++i)
                    for (size_t j = 0; j < cols; ++j)
                        f(values[getIndex(i, j)], i, j);
            }
        }

        constexpr size_type Rows() const noexcept {return rows;}
        constexpr size_type Cols() const noexcept {return cols;}

        static constexpr BaseMatrix Matrix(size_type _rows, size_type _cols) {
            return BaseMatrix(_rows, _cols);
        }

        static constexpr BaseMatrix Vector(size_type _rows) {
            return BaseMatrix(_rows, 1);
        }

        template<typename Typ>
        friend constexpr BaseMatrix<Typ> matmul(const BaseMatrix<Typ>& a, const BaseMatrix<Typ>& b) noexcept;

    private:
        std::vector<value_type> values;
        size_type rows = 1;
        size_type cols = 1;

        constexpr size_type getIndex(size_type _row,size_type _col) const noexcept {
            assert(_row >= 0 && _row < rows);
            assert(_col >= 0 && _col < cols);
            return _row + _col * rows; // column-major
        }
};



template<typename Tp>
constexpr BaseMatrix<Tp> matmul(const BaseMatrix<Tp>& a, const BaseMatrix<Tp>& b) noexcept {
    assert(a.cols == b.rows);

    using Value = BaseMatrix<Tp>::value_type;
    
    BaseMatrix<Tp> result = BaseMatrix<Tp>::Matrix(a.rows, b.cols);

    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t k = 0; k < b.cols; ++k) {
            Value sum = static_cast<Value>(0);

            for (size_t j = 0; j < a.cols; ++j)
                sum += a(i, j) * b(j, k);

            result(i, k) = sum;
        }
    }

    return result;
}


using Matrix  = BaseMatrix<float>;
using dMatrix = BaseMatrix<double>;