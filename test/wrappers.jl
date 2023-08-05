using Imbalance: tablify, matrixify, ERR_TABLE_TYPE

@testset "tablify throws error" begin
    @test_throws ERR_TABLE_TYPE("Int64") begin
        f(x) = x
        tablify(f, 1, [1, 2, 3])
    end
end

@testset "Matrixify" begin
    X = [1.0 2.0; 3.0 4.0]
    df = DataFrame(A = [1.0, 3.0], B = [2.0, 4.0])
    X_matrix, names = matrixify(df)
    @test X_matrix == X
    @test names == Symbol[:A, :B]
end

@testset "tablify with labels provided separately" begin
    X = DataFrame([1.0 2.0; 3.0 4.0], [:A, :B])
    y = [1, 2]

    oversample_func(X, y; kwargs...) = (vcat(X, X), vcat(y, y))
    X_over, y_over = tablify(oversample_func, X, y)
    @test X_over == DataFrame([1.0 2.0; 3.0 4.0; 1.0 2.0; 3.0 4.0], [:A, :B])
    @test y_over == [1, 2, 1, 2]

end

@testset "tablify with labels provided as column index" begin
    Xy = DataFrame([1.0 1 2.0; 3.0 2 4.0], [:A, :Y, :B])
    oversample_func(X, y; kwargs...) = (vcat(X, X), vcat(y, y))
    Xy_over = tablify(oversample_func, Xy, 2)
    @test Xy_over == DataFrame([1.0 1 2.0; 3.0 2 4.0; 1.0 1 2.0; 3.0 2 4.0], [:A, :Y, :B])
end
