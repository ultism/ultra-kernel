import re

F = "dense_blockscaled_gemm_persistent_pingpong.py"
src = open(F).read()

def rep(old, new, n=1):
    global src
    assert old in src, old[:80]
    src = src.replace(old, new, n)

# 1. CLI choices relax
rep('parser.add_argument("--a_major", choices=["k"]', 'parser.add_argument("--a_major", choices=["k", "m"]')
rep('parser.add_argument("--b_major", choices=["k"]', 'parser.add_argument("--b_major", choices=["k", "n"]')
rep('parser.add_argument("--c_major", choices=["n"]', 'parser.add_argument("--c_major", choices=["n", "m"]')

# 2. __init__ flag
rep("""        self.epi_smem_layout_staged = None
""",
"""        self.epi_smem_layout_staged = None

        # 8-bit MN-major B: TMA cannot transpose 8-bit data, so the load warps
        # copy B manually (cp.async, 16B units) into a standard K-major SMEM
        # layout, guarded by an extra PipelineCpAsync next to the TMA pipeline.
        self.b_manual_load = False
""")

# 3. __call__: flag + conditional TMA-B
rep("""        self.c_layout = utils.LayoutEnum.from_tensor(c)

        self._setup_attributes()""",
"""        self.c_layout = utils.LayoutEnum.from_tensor(c)

        self.b_manual_load = (
            self.b_dtype.width == 8 and self.b_layout.is_n_major_b()
        )

        self._setup_attributes()""")

rep("""        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            b,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            1,
            internal_type=self.tma_internal_b_dtype,
        )""",
"""        if cutlass.const_expr(self.b_manual_load):
            # B is copied manually by the load warps; its TMA atom is unused.
            # Build a harmless dummy from A and pass the kernel the raw B.
            tma_atom_b, _ = self._make_tma_atoms_and_tensors(
                a,
                self.a_smem_layout_staged,
                (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
                1,
                internal_type=self.tma_internal_a_dtype,
            )
            tma_tensor_b = b
        else:
            tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
                b,
                self.b_smem_layout_staged,
                (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
                1,
                internal_type=self.tma_internal_b_dtype,
            )""")

# 4. SharedStorage barrier storage
rep("""            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
""",
"""            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            b_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
""")

# 5. tx_count excludes B when manual
rep("""        tma_copy_bytes = (
            cute.size_in_bytes(self.a_dtype, a_smem_layout)
            + cute.size_in_bytes(self.b_dtype, b_smem_layout)
            + cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
            + cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        )""",
"""        tma_copy_bytes = (
            cute.size_in_bytes(self.a_dtype, a_smem_layout)
            + cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
            + cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        )
        if cutlass.const_expr(not self.b_manual_load):
            tma_copy_bytes += cute.size_in_bytes(self.b_dtype, b_smem_layout)""")

# 6. b_pipeline + states + tiled copy creation (after mainloop_pipeline.create)
rep("""            barrier_storage=mainloop_pipeline_array_ptr,
            cta_layout_vmnk=cta_layout_vmnk,
        )
""",
"""            barrier_storage=mainloop_pipeline_array_ptr,
            cta_layout_vmnk=cta_layout_vmnk,
        )
        if cutlass.const_expr(self.b_manual_load):
            # Guards the cp.async B stores: full trips once all 128 load
            # threads' cp.asyncs complete; each stage is released by the 128
            # threads of the consuming MMA warpgroup.
            b_pipeline = pipeline.PipelineCpAsync.create(
                num_stages=self.ab_stage,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, 128
                ),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, self.num_mma_warps // 2 * 32
                ),
                barrier_storage=storage.b_pipeline_array_ptr.data_ptr(),
            )
            b_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.ab_stage
            )
            b_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )
            # 128 load threads x 16 consecutive N' elements: cp.async reads
            # 16B-contiguous gmem N'-runs and writes 16B-contiguous N'-runs in
            # the K-major swizzled SMEM layout (no transpose pass needed).
            tiled_copy_b_manual = cute.make_tiled_copy_tv(
                cute.make_copy_atom(
                    cute.nvgpu.cpasync.CopyG2SOp(), self.b_dtype
                ),
                cute.make_layout((8, 16), stride=(16, self.tile_shape_mnk[1])),
                cute.make_layout(16, stride=1),
            )
""")

# 7. tma_partition B skip
rep("""        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_nkl, 0, 2),
        )""",
"""        if cutlass.const_expr(not self.b_manual_load):
            tBsB, tBgB = cpasync.tma_partition(
                tma_atom_b,
                b_cta_crd,
                b_cta_layout,
                cute.group_modes(sB, 0, 2),
                cute.group_modes(gB_nkl, 0, 2),
            )""")

# 8. producer branch
rep("""            if warp_idx == self.tma_load_warp_id:
                work_tile = tile_sched.initial_work_tile_info()""",
"""            if cutlass.const_expr(self.b_manual_load):
                # Warps 8-11 all run the producer loop: warp 8 issues TMA loads
                # for A/SFA/SFB; all four warps cp.async B into the K-major
                # SMEM stage, transposing on the fly via 16B chunk placement.
                # Pipeline state .advance() stays at loop level (never inside
                # the dynamic warp-if) to keep SSA dominance intact.
                thr_copy_b_manual = tiled_copy_b_manual.get_slice(
                    tidx - self.tma_load_warp_id * 32
                )
                work_tile = tile_sched.initial_work_tile_info()
                while work_tile.is_valid_tile:
                    tile_coord_mnl = work_tile.tile_idx
                    tAgA_mkl = tAgA[
                        (None, tile_coord_mnl[0], None, tile_coord_mnl[2])
                    ]
                    tAgSFA_mkl = tAgSFA[
                        (None, tile_coord_mnl[0], None, tile_coord_mnl[2])
                    ]
                    tBgSFB_nkl = tBgSFB[
                        (None, tile_coord_mnl[1], None, tile_coord_mnl[2])
                    ]
                    gB_nk = gB_nkl[
                        (None, None, tile_coord_mnl[1], None, tile_coord_mnl[2])
                    ]

                    mainloop_producer_state.reset_count()
                    b_producer_state.reset_count()

                    for k_tile in range(0, k_tile_cnt, 1, unroll=1):
                        if warp_idx == self.tma_load_warp_id:
                            mainloop_pipeline.producer_acquire(
                                mainloop_producer_state
                            )
                            tAgA_k = tAgA_mkl[
                                (None, mainloop_producer_state.count)
                            ]
                            tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]
                            tAgSFA_k = tAgSFA_mkl[
                                (None, mainloop_producer_state.count)
                            ]
                            tAsSFA_pipe = tAsSFA[
                                (None, mainloop_producer_state.index)
                            ]
                            tBgSFB_k = tBgSFB_nkl[
                                (None, mainloop_producer_state.count)
                            ]
                            tBsSFB_pipe = tBsSFB[
                                (None, mainloop_producer_state.index)
                            ]
                            cute.copy(
                                tma_atom_a,
                                tAgA_k,
                                tAsA_pipe,
                                tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                    mainloop_producer_state
                                ),
                            )
                            cute.copy(
                                tma_atom_sfa,
                                tAgSFA_k,
                                tAsSFA_pipe,
                                tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                    mainloop_producer_state
                                ),
                            )
                            cute.copy(
                                tma_atom_sfb,
                                tBgSFB_k,
                                tBsSFB_pipe,
                                tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                    mainloop_producer_state
                                ),
                            )
                            mainloop_pipeline.producer_commit(
                                mainloop_producer_state
                            )

                        b_pipeline.producer_acquire(b_producer_state)
                        gB_k = gB_nk[(None, None, b_producer_state.count)]
                        tBgB_man = thr_copy_b_manual.partition_S(gB_k)
                        tBsB_man = thr_copy_b_manual.partition_D(
                            sB[(None, None, b_producer_state.index)]
                        )
                        cute.copy(tiled_copy_b_manual, tBgB_man, tBsB_man)
                        b_pipeline.producer_commit(b_producer_state)
                        b_producer_state.advance()
                        mainloop_producer_state.advance()

                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()
                # end of while loop

                if warp_idx == self.tma_load_warp_id:
                    mainloop_pipeline.producer_tail(mainloop_producer_state)
                b_pipeline.producer_tail(b_producer_state)
            elif warp_idx == self.tma_load_warp_id:
                work_tile = tile_sched.initial_work_tile_info()""")

# 9. consumer additions
rep("""                mainloop_consumer_state.reset_count()

                math_wg_order_barrier.wait(math_wg_order_state)""",
"""                mainloop_consumer_state.reset_count()
                if cutlass.const_expr(self.b_manual_load):
                    b_consumer_state.reset_count()

                math_wg_order_barrier.wait(math_wg_order_state)""")

rep("""                #  Wait for TMA copies to complete
                mainloop_pipeline.consumer_wait(
                    mainloop_consumer_state, peek_ab_full_status
                )
""",
"""                #  Wait for TMA copies to complete
                mainloop_pipeline.consumer_wait(
                    mainloop_consumer_state, peek_ab_full_status
                )
                if cutlass.const_expr(self.b_manual_load):
                    b_pipeline.consumer_wait(b_consumer_state)
""")

rep("""                        if k_block_idx == num_k_blocks - 1:
                            mainloop_pipeline.consumer_release(mainloop_consumer_state)
                            mainloop_consumer_state.advance()
""",
"""                        if k_block_idx == num_k_blocks - 1:
                            mainloop_pipeline.consumer_release(mainloop_consumer_state)
                            mainloop_consumer_state.advance()
                            if cutlass.const_expr(self.b_manual_load):
                                b_pipeline.consumer_release(b_consumer_state)
                                b_consumer_state.advance()
""")

rep("""                            mainloop_pipeline.consumer_wait(
                                mainloop_consumer_state, peek_ab_full_status
                            )
                        # Copy data from smem to tCrA/tCrB for the next k_block""",
"""                            mainloop_pipeline.consumer_wait(
                                mainloop_consumer_state, peek_ab_full_status
                            )
                            if cutlass.const_expr(self.b_manual_load):
                                b_pipeline.consumer_wait(b_consumer_state)
                        # Copy data from smem to tCrA/tCrB for the next k_block""")

rep("""                    if k_block_idx == num_k_blocks - 1:
                        mainloop_pipeline.consumer_release(mainloop_consumer_state)
                        mainloop_consumer_state.advance()

                    if k_block_next > 0:""",
"""                    if k_block_idx == num_k_blocks - 1:
                        mainloop_pipeline.consumer_release(mainloop_consumer_state)
                        mainloop_consumer_state.advance()
                        if cutlass.const_expr(self.b_manual_load):
                            b_pipeline.consumer_release(b_consumer_state)
                            b_consumer_state.advance()

                    if k_block_next > 0:""")

# 10. s2r atom for B: SMEM is K-major under manual load -> no transpose
rep("""            atom_copy_ldmatrix_B = make_ldmatrix_atom(
                self.b_dtype,
                transpose=self.b_layout.is_n_major_b(),
                num_matrices=4,
                mixed_mode=self.mixed_mode,
            )""",
"""            atom_copy_ldmatrix_B = make_ldmatrix_atom(
                self.b_dtype,
                transpose=self.b_layout.is_n_major_b()
                and not self.b_manual_load,
                num_matrices=4,
                mixed_mode=self.mixed_mode,
            )""")

# 11. smem layout: force K-major for manual B (staticmethod -> add param)
rep("""        sf_vec_size: int,
        tiled_mma: cute.TiledMma,
    ) -> tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]:""",
"""        sf_vec_size: int,
        tiled_mma: cute.TiledMma,
        b_manual_load: bool = False,
    ) -> tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]:""")

rep("""            self.epi_stage,
            self.sf_vec_size,
            self.tiled_mma,
        )""",
"""            self.epi_stage,
            self.sf_vec_size,
            self.tiled_mma,
            self.b_manual_load,
        )""")

rep("""        b_smem_shape = cute.slice_(tile_shape_mnk, (0, None, None))

        b_major_mode_size = tile_shape_mnk[2 if b_is_k_major else 1]
        b_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                b_layout,
                b_dtype,
                b_major_mode_size,
            ),
            b_dtype,
        )
        b_smem_layout_staged = cute.tile_to_shape(
            b_smem_layout_atom,
            cute.append(b_smem_shape, ab_stage),
            order=(0, 1, 2) if b_is_k_major else (1, 0, 2),
        )""",
"""        b_smem_shape = cute.slice_(tile_shape_mnk, (0, None, None))

        if b_manual_load:
            # The manual cp.async loader transposes on the fly and always
            # writes a standard K-major SMEM tile.
            b_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
                sm90_utils.get_smem_layout_atom(
                    utils.LayoutEnum.ROW_MAJOR,
                    b_dtype,
                    tile_shape_mnk[2],
                ),
                b_dtype,
            )
            b_smem_layout_staged = cute.tile_to_shape(
                b_smem_layout_atom,
                cute.append(b_smem_shape, ab_stage),
                order=(0, 1, 2),
            )
        else:
            b_major_mode_size = tile_shape_mnk[2 if b_is_k_major else 1]
            b_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
                sm90_utils.get_smem_layout_atom(
                    b_layout,
                    b_dtype,
                    b_major_mode_size,
                ),
                b_dtype,
            )
            b_smem_layout_staged = cute.tile_to_shape(
                b_smem_layout_atom,
                cute.append(b_smem_shape, ab_stage),
                order=(0, 1, 2) if b_is_k_major else (1, 0, 2),
            )""")

# 12. alignment gate for the manual path
rep("""    if not Sm120BlockScaledGemmKernel.is_valid_tensor_alignment(
        m, n, k, l, a_dtype, c_dtype, a_major, b_major, c_major
    ):
        raise ValueError("Invalid tensor alignment")""",
"""    if not Sm120BlockScaledGemmKernel.is_valid_tensor_alignment(
        m, n, k, l, a_dtype, c_dtype, a_major, b_major, c_major
    ):
        raise ValueError("Invalid tensor alignment")

    if b_major == "n" and (n % 128 != 0 or k % 128 != 0):
        raise ValueError("manual MN-major B path requires N%128==0 and K%128==0")""")

open(F, "w").write(src)
print("patched OK")
