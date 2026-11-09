import {
    BindingTwoWay,
    BooleanInput,
    Label,
    LabelGroup,
    Panel,
    SelectInput,
    SliderInput
} from '@playcanvas/pcui/react';

/**
 * @import { Observer } from '@playcanvas/observer'
 * @import { ReactElement } from 'react'
 */

/**
 * @param {object} props - The props.
 * @param {Observer} props.observer - The observer.
 * @returns {ReactElement} The control panel.
 */
export function Controls({ observer }) {
    return (
        <Panel headerText='Jungle'>
            <LabelGroup text='Error px'>
                <SliderInput
                    binding={new BindingTwoWay()}
                    link={{ observer, path: 'data.threshold' }}
                    min={0.25}
                    max={32}
                    precision={2}
                />
            </LabelGroup>
            <LabelGroup text='Impostor px'>
                <SliderInput
                    binding={new BindingTwoWay()}
                    link={{ observer, path: 'data.impostorPx' }}
                    min={0}
                    max={400}
                    precision={0}
                />
            </LabelGroup>
            <LabelGroup text='Impostor bright'>
                <SliderInput
                    binding={new BindingTwoWay()}
                    link={{ observer, path: 'data.impostorBrightness' }}
                    min={0.2}
                    max={1.5}
                    precision={2}
                />
            </LabelGroup>
            <LabelGroup text='Impostor AO'>
                <SliderInput
                    binding={new BindingTwoWay()}
                    link={{ observer, path: 'data.impostorOcclusion' }}
                    min={0}
                    max={0.6}
                    precision={2}
                />
            </LabelGroup>
            <LabelGroup text='TAA'>
                <BooleanInput
                    type='toggle'
                    binding={new BindingTwoWay()}
                    link={{ observer, path: 'data.taa' }}
                />
            </LabelGroup>
            <LabelGroup text='Impostor tint'>
                <BooleanInput
                    type='toggle'
                    binding={new BindingTwoWay()}
                    link={{ observer, path: 'data.impostorTint' }}
                />
            </LabelGroup>
            <LabelGroup text='Drawn'>
                <Label
                    binding={new BindingTwoWay()}
                    link={{ observer, path: 'data.stats' }}
                    value={observer.get('data.stats')}
                />
            </LabelGroup>
            <LabelGroup text='Occlusion culling'>
                <BooleanInput
                    type='toggle'
                    binding={new BindingTwoWay()}
                    link={{ observer, path: 'data.occlusion' }}
                />
            </LabelGroup>
            <LabelGroup text='Sun'>
                <SliderInput
                    binding={new BindingTwoWay()}
                    link={{ observer, path: 'data.sun' }}
                    min={0}
                    max={10}
                    precision={2}
                />
            </LabelGroup>
            <LabelGroup text='Sky'>
                <SliderInput
                    binding={new BindingTwoWay()}
                    link={{ observer, path: 'data.sky' }}
                    min={0}
                    max={2}
                    precision={2}
                />
            </LabelGroup>
            <LabelGroup text='Exposure'>
                <SliderInput
                    binding={new BindingTwoWay()}
                    link={{ observer, path: 'data.exposure' }}
                    min={0.1}
                    max={2}
                    precision={2}
                />
            </LabelGroup>
            <LabelGroup text='Canopy density'>
                <SliderInput
                    binding={new BindingTwoWay()}
                    link={{ observer, path: 'data.canopyK' }}
                    min={0}
                    max={4}
                    precision={2}
                />
            </LabelGroup>
            <LabelGroup text='Canopy sun'>
                <SliderInput
                    binding={new BindingTwoWay()}
                    link={{ observer, path: 'data.canopySun' }}
                    min={0}
                    max={1}
                    precision={2}
                />
            </LabelGroup>
            <LabelGroup text='Canopy sky'>
                <SliderInput
                    binding={new BindingTwoWay()}
                    link={{ observer, path: 'data.canopySky' }}
                    min={0}
                    max={1}
                    precision={2}
                />
            </LabelGroup>
            <LabelGroup text='Visualisation'>
                <SelectInput
                    type='string'
                    binding={new BindingTwoWay()}
                    link={{ observer, path: 'data.visualization' }}
                    options={[
                        { v: 'material', t: 'Material' },
                        { v: 'meshlet', t: 'Meshlet colours' },
                        { v: 'lod', t: 'LOD tier' },
                        { v: 'normals', t: 'World normals' }
                    ]}
                />
            </LabelGroup>
        </Panel>
    );
}
