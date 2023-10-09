using Imbalance: checkbalance, generate_imbalanced_data

@testset "testing checkbalance" begin
    num_rows = 100
    num_cont_feats = 4
    class_probs = [0.9, 0.1]
    X, y = generate_imbalanced_data(
        num_rows,
        num_cont_feats;
        class_probs,
        rng = Random.MersenneTwister(42),
    )
    c = IOCapture.capture() do
        return checkbalance(y)
    end
    @test c.output ==
          """1: ▇▇▇▇▇▇ 10 (11.1%) \n0: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 90 (100.0%) \n"""
end
