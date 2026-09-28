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
        <Panel headerText='Caldera'>
            <LabelGroup text='Jump to'>
                <SelectInput
                    type='string'
                    binding={new BindingTwoWay()}
                    link={{ observer, path: 'data.region' }}
                    options={[
                        { v: 'capital', t: 'Capital (incl. restaurant)' },
                        { v: 'airfield', t: 'Airfield' },
                        { v: 'phosphate_mine', t: 'Phosphate Mine' },
                        { v: 'beachhead', t: 'Beachhead' },
                        { v: 'tile_n', t: 'Tile N (power station)' },
                        { v: 'tile_p', t: 'Tile P (hotel)' },
                        { v: 'arsenal', t: 'Arsenal' },
                        { v: 'tile_f', t: 'Tile F' },
                        { v: 'docks', t: 'Docks' },
                        { v: 'caldera', t: 'Caldera' },
                        { v: 'subpen', t: 'Sub Pen' },
                        { v: 'hotel', t: 'Hotel' },
                        { v: 'power_station', t: 'Power Station' },
                        { v: 'terrain', t: 'Whole island' }
                    ]}
                />
            </LabelGroup>
            <LabelGroup text='Error px'>
                <SliderInput
                    binding={new BindingTwoWay()}
                    link={{ observer, path: 'data.threshold' }}
                    min={0.25}
                    max={32}
                    precision={2}
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
            <LabelGroup text='Collision'>
                <BooleanInput
                    type='toggle'
                    binding={new BindingTwoWay()}
                    link={{ observer, path: 'data.collision' }}
                />
            </LabelGroup>
            <LabelGroup text='Collision class'>
                <SelectInput
                    type='string'
                    binding={new BindingTwoWay()}
                    link={{ observer, path: 'data.collisionClass' }}
                    options={[
                        { v: 'all', t: 'All' },
                        { v: 'player_clip', t: 'Player clip' },
                        { v: 'weapon_clip', t: 'Weapon clip' },
                        { v: 'world', t: 'World geometry' }
                    ]}
                />
            </LabelGroup>
        </Panel>
    );
}
