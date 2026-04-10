using Test

include("../src/Lattice/Square.jl")

@testset "SquareLattice" begin

    @testset "1D chain (Lx=4, Ly=1, OBC)" begin
        lat = SquareLattice(4, 1)
        @test length(lat) == 4
        @test lat.Lx == 4
        @test lat.Ly == 1

        # site_index: 1-indexed
        @test site_index(lat, 1, 1) == 1
        @test site_index(lat, 2, 1) == 2
        @test site_index(lat, 4, 1) == 4

        # site_coord round-trip
        for i in 1:4
            x, y = site_coord(lat, i)
            @test site_index(lat, x, y) == i
        end

        # bonds: nearest-neighbor pairs (1,2),(2,3),(3,4)
        b = bonds(lat)
        @test length(b) == 3
        @test (1,2) in b
        @test (2,3) in b
        @test (3,4) in b
        @test all(i < j for (i,j) in b)   # sorted, i < j
    end

    @testset "1D chain PBC (Lx=4, Ly=1, xpbc)" begin
        lat = SquareLattice(4, 1; xpbc=true)
        b = bonds(lat)
        @test length(b) == 4
        @test (1,2) in b
        @test (2,3) in b
        @test (3,4) in b
        @test (1,4) in b   # wraparound bond
        @test all(i < j for (i,j) in b)
    end

    @testset "2D lattice (Lx=3, Ly=2, OBC)" begin
        lat = SquareLattice(3, 2)
        # Sites: (1,1)=1, (2,1)=2, (3,1)=3, (1,2)=4, (2,2)=5, (3,2)=6
        @test length(lat) == 6
        @test site_index(lat, 1, 1) == 1
        @test site_index(lat, 3, 2) == 6

        b = bonds(lat)
        # x-bonds: (1,2),(2,3),(4,5),(5,6)   (4 bonds)
        # y-bonds: (1,4),(2,5),(3,6)          (3 bonds)
        @test length(b) == 7
        @test (1,2) in b
        @test (2,3) in b
        @test (4,5) in b
        @test (5,6) in b
        @test (1,4) in b
        @test (2,5) in b
        @test (3,6) in b
    end

    @testset "2D lattice PBC" begin
        lat = SquareLattice(3, 2; xpbc=true, ypbc=true)
        b = bonds(lat)
        # x-bonds row 1: (1,2),(2,3),(1,3) wraparound = 3
        # x-bonds row 2: (4,5),(5,6),(4,6) wraparound = 3
        # y-bonds: (1,4),(2,5),(3,6) = 3
        # Total: 3 + 3 + 3 = 9
        @test length(b) == 9
        @test all(i < j for (i,j) in b)
    end

    @testset "error on bad dimensions" begin
        @test_throws ErrorException SquareLattice(0, 3)
        @test_throws ErrorException SquareLattice(3, 0)
    end

    @testset "show" begin
        lat = SquareLattice(3, 2; xpbc=true)
        s = repr(lat)
        @test occursin("Lx=3", s)
        @test occursin("Ly=2", s)
        @test occursin("xpbc", s)
    end
end
