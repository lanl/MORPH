import asyncio

import torch

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from nomad.tensor_codac import serialize_tensor, deserialize_tensor

from morph_pde.nomad_tool import MORPHTool


async def main():
    torch.manual_seed(0)

    # -------------------------------------------------
    # 1. Create exactly one input
    # Shape: (T, F, C, D, H, W)
    # -------------------------------------------------
    x = torch.randn(1, 1, 1, 1, 64, 64)

    print("Input shape:", tuple(x.shape))

    # -------------------------------------------------
    # 2. Run MORPH directly
    # -------------------------------------------------
    
    direct_tool = MORPHTool.from_pretrained(
        r"models\morph-ti-fm",
        device=torch.device("cuda:0"),
    )

    with torch.no_grad():
        direct_prediction = direct_tool.fm(
            x.unsqueeze(0).to("cuda:0")
        )[2][0].detach().cpu()

    print(
        "Direct output shape:",
        tuple(direct_prediction.shape),
    )

    # -------------------------------------------------
    # 3. Send EXACT SAME tensor through MCP
    # -------------------------------------------------
    encoded_state = serialize_tensor(x)

    async with streamable_http_client(
        "http://localhost:8000/mcp"
    ) as streams:

        read_stream = streams[0]
        write_stream = streams[1]

        async with ClientSession(
            read_stream,
            write_stream,
        ) as session:

            await session.initialize()

            tools = await session.list_tools()

            print(
                "Available tools:",
                [tool.name for tool in tools.tools],
            )

            result = await session.call_tool(
                "morph_ti",
                {"state": encoded_state},
            )

            result_dict = result.model_dump(
                by_alias=True
            )

            is_error = result_dict.get(
                "isError",
                result_dict.get("is_error", False),
            )

            print("MCP error:", is_error)

            if is_error:
                print(result_dict)
                return

            structured = result_dict.get(
                "structuredContent",
                result_dict.get(
                    "structured_content"
                ),
            )

            mcp_prediction = deserialize_tensor(
                structured["prediction"]
            ).cpu()

    # -------------------------------------------------
    # 4. Compare direct and MCP predictions
    # -------------------------------------------------
    print(
        "MCP output shape:",
        tuple(mcp_prediction.shape),
    )

    difference = (
        direct_prediction - mcp_prediction
    ).abs()

    print(
        "Max abs diff:",
        difference.max().item(),
    )

    print(
        "Mean abs diff:",
        difference.mean().item(),
    )

    match = torch.allclose(
        direct_prediction,
        mcp_prediction,
        atol=1e-5,
        rtol=1e-5,
    )

    print("Match:", match)


if __name__ == "__main__":
    asyncio.run(main())