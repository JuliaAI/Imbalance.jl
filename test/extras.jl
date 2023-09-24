using Imbalance: checkbalance, generate_imbalanced_data

@testset "testing checkbalance" begin
    num_rows = 100
    num_cont_feats = 4
    probs = [0.9, 0.1]
    X, y = generate_imbalanced_data(
        num_rows,
        num_cont_feats;
        probs,
        rng = 42,
    )
    c = IOCapture.capture() do
        return checkbalance(y)
    end
    @test c.output ==
          """1: ▇▇▇▇▇▇▇▇▇▇ 16 (19.0%) \n0: ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 84 (100.0%) \n"""
end
